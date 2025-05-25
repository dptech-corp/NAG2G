from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.modules import LayerNorm

class FPClassfierHead(nn.Module):

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_layer,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.dense2 = nn.Linear(inner_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.bn = nn.BatchNorm1d(inner_dim)
        self.num_layer = num_layer

    def forward(self, x, **kwargs):
        
        x = self.dense1(x)
        for i in range(self.num_layer):
            x_res = x
            x = self.dropout(x)
            x = self.dense2(x)
            x = self.bn(x)
            x = self.activation_fn(x)
            x = x + x_res

        x = self.out_proj(x)
        x_class = F.softmax(x, dim=-1)
        return x_class


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, padding_mask, **kwargs):
        x = features[:, 0, :].clone() # take <s> token (equiv. to [CLS])
        # if padding_mask is not None:
        #     padding_mask_x = 1 - padding_mask.type_as(features)
        #     reverse_len = 1/(padding_mask_x.sum(1))
        #     sum_o = features.sum(1)
        #     x = sum_o * (reverse_len.unsqueeze(1))
        # else:
        #     x = features.mean(1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class LayerAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        scaling_factor=1,
        bias_flag = True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim*3, bias = bias_flag)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.dropout_rate = pooler_dropout
        self.dense = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, num_classes)
        self.scaling = (input_dim * scaling_factor) ** -0.5

    def forward(self, features, padding_mask, training_flag, **kwargs):
        
 
        q,k,v = self.input_proj(features).chunk(3, dim=-1)
        q = q * self.scaling
        attn_weights = torch.bmm(q, k.transpose(1, 2).clone())

        if padding_mask is not None:
            # don't attend to padding symbols
            attn_weights.masked_fill_(
                padding_mask.unsqueeze(1).to(torch.bool), float("-inf")
            )

        attn = softmax_dropout(attn_weights, self.dropout_rate, training_flag, inplace = False)
        o = torch.bmm(attn, v)
        x = o[:, 0, :]
        if padding_mask is not None:
            padding_mask_x = 1 - padding_mask.type_as(features)
            reverse_len = 1/(padding_mask_x.sum(1))
            sum_o = o.sum(1)
            x = sum_o * (reverse_len.unsqueeze(1))
            x = x
        else:
            x = o.mean(1)

        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)        
        return x