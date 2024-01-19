# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import torch
import numpy as np
from functools import lru_cache
import logging
from unicore.data import data_utils, BaseWrapperDataset

logger = logging.getLogger(__name__)


class ReorderDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, molstr, atoms, coordinates, nomalize=True):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.molstr = molstr
        self.coordinates = coordinates
        self.nomalize = nomalize
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        # print('???',self.molstr )
        if self.molstr == 'smi':
            molstr = np.array([x for x in self.dataset[index][self.molstr]])
        elif self.molstr == 'selfies':
            molstr = np.array(self.dataset[index][self.molstr])
        # else:
        #     atoms = np.array(self.dataset[index][self.atoms])
        receptor = self.dataset[index]['receptor']
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0, (len(atoms), atoms, self.atoms, molstr)
        # print('???',len(atoms))
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        if len(atoms) > 256:
            np.random.seed(self.seed)
            index = np.random.choice(len(atoms), 256, replace=False)
            atoms = atoms[index]
            coordinates = coordinates[index]
        # normalize
        if self.nomalize:
            coordinates = coordinates - coordinates.mean(axis=0)
        #print( self.atoms,len(atoms), self.molstr, len(molstr))
        return {self.atoms: atoms, self.molstr: molstr, 'coordinates': coordinates.astype(np.float32), 'receptor': receptor}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
