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
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi
            return Chem.MolToSmiles(mol, doRandom=True)
        except:
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
        tmp = np.array([atoms_map_reactant_dict[i] for i in atoms_map_product])
        orders = np.array([i for i in range(len(atoms_map_reactant))])
        mask = np.array(atoms_map_reactant) != 0
        list_reactant = np.concatenate([tmp, orders[~mask]], 0).tolist()
        return list_reactant
    
    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        product = self.dataset[index]
        reactant = self.reactant_dataset[index]
        product_map, _ = self.get_map(product)
        reactant_map, reactant_mol = self.get_map(reactant)
        list_reactant = self.get_list(product_map, reactant_map)
        nm = Chem.RenumberAtoms(reactant_mol, list_reactant)
        return Chem.MolToSmiles(nm, canonical=False)
