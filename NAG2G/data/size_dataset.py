# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import logging
logger = logging.getLogger(__name__)


class SizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, src_key_info, tgt_key_info):
        self.dataset = dataset
        self.src_size = np.array([])
        self.tgt_size = np.array([])
        self.cal_size_data(dataset, src_key_info, tgt_key_info)

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx]

    def cal_size_data(self, dataset, src_key_info, tgt_key_info):
        for i in range(len(dataset)):
            self.src_size = np.append(
                self.src_size, len(dataset[i][src_key_info]))
            self.tgt_size = np.append(
                self.tgt_size, len(dataset[i][tgt_key_info]))
            if i % 10000 == 0:
                print('test dataset size: ', i)

    def get_size_data(self):
        return self.src_size, self.tgt_size
