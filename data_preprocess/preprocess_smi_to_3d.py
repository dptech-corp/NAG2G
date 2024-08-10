import numpy as np
import warnings
import contextlib
import timeout_decorator
from sklearn.mixture import BayesianGaussianMixture

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")


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
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
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
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]
    )


@timeout_decorator.timeout(30)
def inner_smi2coords(smi, num_confs=100, seed=42, cluster_size=10):
    coordinate_list, rotable_bonds_list = [], []
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol, addCoords=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    wt = Descriptors.ExactMolWt(mol)
    # skip for heavy molecules
    if wt > 2000:
        return None
    # at least have two atoms
    if len(atoms) < 2:
        return None

    # allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=1)
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)

    if res == 0:
        rotable_bonds = get_torsions(mol)
        for i in range(num_confs):
            np.random.seed(i)
            values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
            for idx in range(len(rotable_bonds)):
                SetDihedral(mol.GetConformer(), rotable_bonds[idx], values[idx])
            Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer())
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                coordinate_list.append(
                    mol.GetConformer().GetPositions().astype(np.float32)
                )
                rotable_bonds_value = [
                    GetDihedral(mol.GetConformer(), rotable_bonds[idx])
                    for idx in range(len(rotable_bonds))
                ]
                rotable_bonds_list.append(rotable_bonds_value)
            except:
                continue

    elif res == -1:
        mol_tmp = Chem.MolFromSmiles(smi)
        AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
        rotable_bonds = get_torsions(mol_tmp)
        for i in range(num_confs):
            np.random.seed(i)
            values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
            for idx in range(len(rotable_bonds)):
                SetDihedral(mol_tmp.GetConformer(), rotable_bonds[idx], values[idx])
            Chem.rdMolTransforms.CanonicalizeConformer(mol_tmp.GetConformer())
            try:
                AllChem.MMFFOptimizeMolecule(mol_tmp)
                coordinate_list.append(
                    mol_tmp.GetConformer().GetPositions().astype(np.float32)
                )
                rotable_bonds_value = [
                    GetDihedral(mol_tmp.GetConformer(), rotable_bonds[idx])
                    for idx in range(len(rotable_bonds))
                ]
                rotable_bonds_list.append(rotable_bonds_value)
            except:
                continue
    if num_confs != cluster_size:
        X = np.array(rotable_bonds_list)
        clf = BayesianGaussianMixture(n_components=cluster_size, random_state=seed).fit(
            X
        )
        probs = clf.predict_proba(X)
        # filter redundant clusters
        probs = probs[:, probs.mean(axis=0) != 0.0]
        ids = probs.argmax(axis=0)
        # padding to cluster_size
        if len(ids) < cluster_size:
            ids = ids + [ids[0]] * (cluster_size - len(ids))
        cluster_coordinate_list = [coordinate_list[idx] for idx in ids]
        print(ids)
    else:
        cluster_coordinate_list = coordinate_list
    return {"atoms": atoms, "coordinates": cluster_coordinate_list, "smi": smi}


def smi2coords_3D(smi):
    try:
        return inner_smi2coords(smi, num_confs=10)
    except:
        return None


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2coords_2D(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        coordinate_list = [smi2_2Dcoords(smi).astype(np.float32)]
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
        return {"atoms": atoms, "coordinates": coordinate_list, "smi": smi}
    except:
        return None
