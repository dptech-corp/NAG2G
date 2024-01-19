import os
import sys
import json
import glob
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
import contextlib
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
import timeout_decorator
from scipy.spatial.transform import Rotation
from sklearn.mixture import BayesianGaussianMixture
from rdkit.Chem import rdMolTransforms
import copy

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_torsions(m):
    m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if (b1.GetIdx() == bond.GetIdx()):
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if ((b2.GetIdx() == bond.GetIdx())
                    or (b2.GetIdx() == b1.GetIdx())):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if (idx4 == idx1):
                    continue
                # skip torsions that include hydrogens
                if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    continue    
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break   
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
           
def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

@timeout_decorator.timeout(20)
def inner_smi2coords(smi, num_confs=100, seed=42):
    coordinate_list, target_coordinate_list = [], []
    can_smi = Chem.CanonSmiles(smi)
    mol = Chem.MolFromSmiles(can_smi)
    AllChem.AddHs(mol, addCoords=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    wt = Descriptors.ExactMolWt(mol)
    # skip for heavy molecules
    if wt > 1000:
        return None
    # at least have two atoms
    if len(atoms) < 2:
        return None

    # allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=1)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    rotable_bonds = get_torsions(mol)
    for i in range(num_confs):
        np.random.seed(i)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(mol.GetConformer(), rotable_bonds[idx], values[idx])
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer())
        try:
            coordinate_list.append(mol.GetConformer().GetPositions().astype(np.float32))
            AllChem.MMFFOptimizeMolecule(mol)
            target_coordinate_list.append(mol.GetConformer().GetPositions().astype(np.float32))
        except:
            continue

    return pickle.dumps({'atoms': atoms, 
                        'coordinates': coordinate_list,
                        'target_coordinates': target_coordinate_list,
                        'can_smi': can_smi,
                        'id': smi, }, protocol=-1)

def smi2coords(smi):
    try:
        return inner_smi2coords(smi)
    except:
        return None

def get_train_val(smi_list, val_size=100000):
    with numpy_seed(42):
        val_smi = np.random.choice(smi_list, replace=False, size=val_size)
        np.random.shuffle(val_smi)
        train_smi = list(set(smi_list) - set(val_smi))
        np.random.shuffle(train_smi)
    return train_smi, val_smi

def write_lmdb(outpath='.', nthreads=16, nrows=1000000):

    small_size = 10000
    val_size = min(10000, int(nrows*0.01))
    smi_list = pd.read_csv('../molecules/clean_smi.csv.gz', names=['smi'], nrows=nrows)['smi'].tolist()
    print('original size: ', len(smi_list))
    train_smi, val_smi = get_train_val(smi_list, val_size=val_size)
    print('train size: {}; val size: {}'.format(len(train_smi), len(val_smi)))
    task_list = [('valid.lmdb', val_smi),  \
                ('train.small.lmdb', train_smi[:small_size]), \
                ('train.lmdb', train_smi), \
                ]
    for name, smi_list in task_list:
        outputfilename = os.path.join(outpath, name)
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(smi2coords, smi_list), total=len(smi_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
                    if i % 10000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()

def write_v2(lmdb_inpath, lmdb_outpath):
    env = lmdb.open(
        lmdb_inpath,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))

    env_new = lmdb.open(
        lmdb_outpath,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    i = 0 
    for idx in tqdm(range(len(_keys))):
        datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        if len(data['coordinates']) > 0:
            if len(data['coordinates']) == len(data['target_coordinates']):
                txn_write.put(f'{i}'.encode("ascii"), pickle.dumps(data, protocol=-1))
                i += 1
            else:
                print('miss shape size: ', data['id'])
        else:
            print('miss size: ', data['id'])
        if i % 10000 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)
    txn_write.commit()
    env_new.close()
    env.close()

def check(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))
    cnt = 0
    for idx in tqdm(range(len(_keys))):
        datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        # assert len(data["coordinates"]) == len(data["target_coordinates"]), print(data['id'], len(data["coordinates"]), len(data["target_coordinates"]))
        if len(data['coordinates']) != len(data['target_coordinates']):
            cnt += 1
    print(cnt)
    env.close()

if __name__ == '__main__':
    # smi = 'CC(=O)c1ccc2c(c1)N(c3ccccc3S2)CCCN4CCN(CC4)CCO'
    # inner_smi2coords(smi, num_confs=1000, seed=42, cluster_size=10)
    # write_lmdb(outpath='./', nthreads=60, nrows=1000000)
    write_v2(lmdb_inpath='./train.lmdb', lmdb_outpath='./train.v2.lmdb')
    write_v2(lmdb_inpath='./valid.lmdb', lmdb_outpath='./valid.v2.lmdb')
    write_v2(lmdb_inpath='./train.small.lmdb', lmdb_outpath='./train.small.v2.lmdb')
    # check('./train.lmdb')