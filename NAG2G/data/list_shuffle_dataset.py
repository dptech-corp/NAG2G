# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
import random

import logging
logger = logging.getLogger(__name__)


class ListShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, prob=1.0):
        self.dataset = dataset
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        tmp_list = self.dataset[idx]
        if self.prob != 0 and random.random() < self.prob:
            random.shuffle(tmp_list)
        return tmp_list
