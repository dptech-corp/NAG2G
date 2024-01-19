# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from transformers import BartTokenizerFast

import logging
logger = logging.getLogger(__name__)


class BartTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, dict_path, max_seq_len: int = 512):
        self.dataset = dataset
        self.tokenizer = BartTokenizerFast.from_pretrained(dict_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        tmp = self.dataset[idx]
        output = self.tokenizer(tmp)["input_ids"]
        assert len(output) < self.max_seq_len and len(output) > 2
        return torch.LongTensor(output)
