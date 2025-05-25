import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils

def transfer_data_mask(x, token_dict):
    if x in token_dict.keys():
        return True
    else:
        return False 

def transfer_data_id(x, token_dict, common_id):
    if x in token_dict.keys():
        return token_dict[x]
    else:
        return token_dict[common_id]

class SpecialTokenMask():

    def __init__(
        token_dict,
        padding_id,
        common_id
    ):
        super().__init__()
        self.token_dict = token_dict
        self.padding_id = padding_id
        self.common_id = common_id

    def special_token_mask(self, src_tokens):
        src_tokens_data = src_tokens.cpu().numpy()
        apply(transfer_data_mask, src_tokens_data, self.token_dict)
        # src_tokens_ = torch.from_numpy(src_tokens_data)

        return token_mask

    def special_token_transform(self, src_tokens):
        src_tokens_data = src_tokens.cpu().numpy()
        map(src_tokens_data)

        return token_mask_transform_id

 