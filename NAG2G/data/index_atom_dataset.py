import numpy as np
import torch

from functools import lru_cache
from unicore.data import BaseWrapperDataset

from rdkit import Chem


class IndexAtomDataset(BaseWrapperDataset):
    def __init__(
        self,
        smi_dataset: torch.utils.data.Dataset,
    ):
        super().__init__(smi_dataset)
        self.smi_dataset = smi_dataset
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        smiles = self.smi_dataset[index]
        mol = Chem.MolFromSmiles(smiles)
        atom_index = [
            atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"
        ]
        return np.array(atom_index)
