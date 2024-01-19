# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from scipy.spatial import distance_matrix
from functools import lru_cache
from torch.utils.data.dataloader import default_collate
from unicore.data import data_utils, UnicoreDataset


class RandomLabelDataset(UnicoreDataset):
    def __init__(self, maxv, nrow):
        super().__init__()
        self.maxv = maxv
        self.nrow = nrow

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return np.random.randint(self.maxv)

    def __len__(self):
        return self.nrow

    def collater(self, samples):
        return torch.tensor(samples)


class RandomDataset(UnicoreDataset):

    def __init__(self, ncol, nrow, maxv):
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.maxv = maxv

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        with data_utils.numpy_seed(index):
            size = np.random.randint(self.ncol // 2 + 1, self.ncol)
            val = np.random.randint(self.maxv, size=size)
            return torch.tensor(val).long()

    def __len__(self):
        return self.nrow

    def collater(self, samples):
        return default_collate(samples)
