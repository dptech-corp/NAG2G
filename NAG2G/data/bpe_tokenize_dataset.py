import os
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from tokenizers import Tokenizer
from tokenizers.models import BPE

import logging

logger = logging.getLogger(__name__)


def tostring(tmp, shift=30000):
    tmp = tmp.detach().cpu().numpy().tolist()
    return "".join([chr(i + shift) for i in tmp])


def tostringaftsep2(tmp_origin, sep2_ids, shift=30000):
    tmp = tmp_origin.detach().cpu().numpy().tolist()
    if sep2_ids not in tmp:
        return tmp_origin, None
    else:
        idx = tmp.index(sep2_ids)
        if len(tmp) - 1 == idx:
            # TODO: do not support using both sep2 and bpe token
            return tmp_origin[:-1], None
        else:
            return tmp_origin[:idx], "".join([chr(i + shift) for i in tmp[idx + 1 :]])


class BpeTokenizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, tokenizer_path, flag_aftsep2):
        super().__init__(dataset)
        self.tokenizer = Tokenizer(
            BPE.from_file(
                os.path.join(tokenizer_path, "vocab.json"),
                os.path.join(tokenizer_path, "merges.txt"),
            )
        )
        self.flag_aftsep2 = flag_aftsep2
        self.sep2_token_id = self.tokenizer.encode("瘭").ids[0]
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        if self.flag_aftsep2:
            return self.forward_aftsep2(index)
        else:
            return self.forward_normal(index)

    def forward_normal(self, index):
        idx = self.dataset[index]
        result = tostring(idx)
        result = self.tokenizer.encode(result).ids
        result = torch.tensor(result).long()
        return result

    def forward_aftsep2(self, index):
        idx = self.dataset[index]
        origin_tensor, result = tostringaftsep2(idx, self.sep2_token_id)
        if result is None:
            return origin_tensor
        result = self.tokenizer.encode(result).ids
        result = torch.tensor(result).long()
        return torch.cat([origin_tensor, result], 0)
