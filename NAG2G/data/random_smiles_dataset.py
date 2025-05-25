# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import NonCallableMagicMock
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RandomSmilesDataset(BaseWrapperDataset):
    def __init__(self, dataset, prob=1.0):
        super().__init__(dataset)
        self.prob = prob

    def get_random_smiles(self, smi):
        if self.prob == 0 or random.random() >= self.prob:
            return smi
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        for i in range(5):
            smiles = Chem.MolToSmiles(mol, doRandom=True)
            if Chem.MolFromSmiles(smiles) is not None:
                return smiles
            print("RandomSmilesDataset doRandom Fail", i, smiles)
        print("RandomSmilesDataset doRandom Fail", smi)
        return smi

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        result = self.dataset[idx]
        if isinstance(result, list):
            smi = ".".join([self.get_random_smiles(i) for i in self.dataset[idx]])
        elif isinstance(result, str):
            smi = [self.get_random_smiles(i) for i in result.split(".") if i != ""]
            random.shuffle(smi)
            smi = ".".join(smi)
        else:
            raise
        return smi


class ReorderSmilesDataset(BaseWrapperDataset):
    def __init__(self, product_dataset, reactant_dataset):
        super().__init__(product_dataset)
        self.reactant_dataset = reactant_dataset
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.reactant_dataset.set_epoch(epoch)
        self.epoch = epoch

    def get_map(self, smi):
        c_mol = Chem.MolFromSmiles(smi)
        c_id_list = [atom.GetAtomMapNum() for atom in c_mol.GetAtoms()]
        return c_id_list, c_mol

    def get_list(self, atoms_map_product, atoms_map_reactant):
        atoms_map_reactant_dict = {
            atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
        }
        tmp = [
            atoms_map_reactant_dict[i]
            for i in atoms_map_product
            if i in atoms_map_reactant_dict
        ]
        all_indices = set(range(len(atoms_map_reactant)))
        missing_indices = list(all_indices - set(tmp))
        list_reactant = tmp + missing_indices
        return list_reactant

    def get_new_smiles(self, product, reactant):
        product_map, _ = self.get_map(product)
        product_map = [i for i in product_map if i != 0]
        reactant_map, reactant_mol = self.get_map(reactant)
        list_reactant = self.get_list(product_map, reactant_map)
        nm = Chem.RenumberAtoms(reactant_mol, list_reactant)
        new_smiles = Chem.MolToSmiles(nm, canonical=False)
        return new_smiles, nm

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        product = self.dataset[index]
        reactant = self.reactant_dataset[index]
        new_smiles, new_mol = self.get_new_smiles(product, reactant)
        try:
            new_smiles, new_mol = self.get_new_smiles(product, new_smiles)
        except Exception as e:
            print(e)
        if Chem.MolFromSmiles(new_smiles) is None:
            print("ReorderSmilesDataset Fail", reactant, new_smiles)
        return {"smiles": new_smiles, "mol": new_mol}
