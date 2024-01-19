import gzip
import os, sys
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
import numpy as np

lines = gzip.open("data.csv.gz", "r").readlines()

target = []
smiles = []

for i in range(1, len(lines)):
    try:
        s = lines[i].decode().split(",")
        smiles.append(s[1])
        target.append(float(s[2]))
    except:
        target.append(None)

del lines


label_env = lmdb.open(
    "label_3D.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)

with label_env.begin() as txn:
    train_keys = list(txn.cursor().iternext(values=False))


def get_by_key(env, key):
    data = env.begin().get(key)
    if data is None:
        return data
    else:
        try:
            return pickle.loads(gzip.decompress(data))
        except:
            return None


def process_one(key):
    index = int.from_bytes(key, "big")
    label_str = get_by_key(label_env, key)
    label_mol = Chem.MolFromMolBlock(label_str)
    label_mol = Chem.RemoveHs(label_mol)
    ori_smi = Chem.MolToSmiles(Chem.SmilesToMol(smiles[index]))
    label_smi = Chem.MolToSmiles(label_mol)
    if ori_smi != label_smi:
        print("smi mismatch", ori_smi, label_smi)
        return 1
    else:
        return 0


i = 0
erorr_cnt = 0
with Pool(96) as pool:
    for ret in tqdm(pool.imap(process_one, train_keys), total=len(train_keys)):
        erorr_cnt += ret
        # use `int.from_bytes(key, "big")` to decode from bytes
        i += 1


print(erorr_cnt, i)
