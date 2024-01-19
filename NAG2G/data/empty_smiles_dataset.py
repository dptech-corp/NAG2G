import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import re
import pickle
from functools import lru_cache
from unicore.data import UnicoreDataset

logger = logging.getLogger(__name__)


def get_atom(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    return atoms


def get_coordinates(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    try:
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)

        elif res == -1:
            mol_tmp = Chem.MolFromSmiles(smiles)
            AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
            mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
            AllChem.MMFFOptimizeMolecule(mol_tmp)
            coordinates = mol_tmp.GetConformer().GetPositions().astype(np.float32)

    except:
        AllChem.Compute2DCoords(mol)
        coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smiles)
    return [coordinates]


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)
    return tokens


class EMPTY_SMILES_Dataset(UnicoreDataset):
    def __init__(self, init_values, seed=42):
        self.key = []
        self.epoch = 0
        self.put_smiles_in(smiles=init_values, seed=seed)

    def set_epoch(self, epoch, **unused):
        pass

    def put_smiles_in(self, smiles, seed=42):
        self.epoch = (self.epoch + 1) % 100000
        dict_ = {"target_id": smiles}
        dict_["target_atoms"] = get_atom(smiles)
        dict_["target_coordinates"] = get_coordinates(smiles, seed=seed)
        dict_["smiles_target_list"] = [smiles]
        dict_["smiles_target"] = smi_tokenizer(smiles)
        self.key = [dict_]

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.key[index]


class EMPTY_SMILES_Dataset_G2G(UnicoreDataset):
    def __init__(self, name, init_values=None, seed=42):
        self.name = name
        self.key = []
        self.epoch = 0
        self.put_smiles_in(init_values)

    def set_epoch(self, epoch, **unused):
        pass

    def put_smiles_in(self, smiles):
        if smiles is None:
            return
        self.epoch = (self.epoch + 1) % 100000
        self.key = smiles

    def __len__(self):
        return len(self.key)

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return {self.name: self.key[index]}
