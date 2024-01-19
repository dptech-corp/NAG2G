import imp
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from unicore.modules import LayerNorm

from .layers import (
    TransformerEncoderLayer,
    Dropout,
)


class UniMolv2Encoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        pair_dim: int = 64,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        droppath_prob: float = 0.0,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_attention_heads
        self.layer_norm = LayerNorm(embedding_dim)
        self.pair_layer_norm = LayerNorm(pair_dim)
        self.layers = nn.ModuleList([])

        if droppath_prob > 0:
            droppath_probs = [
                x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
            ]
        else:
            droppath_probs = None

        self.layers.extend(
            [
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    pair_dim=pair_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    droppath_prob=droppath_probs[i]
                    if droppath_probs is not None
                    else 0,
                )
                for i in range(num_encoder_layers)
            ]
        )

    def forward(
        self,
        x,
        pair,
        atom_mask,
        pair_mask,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.layer_norm(x)
        pair = self.pair_layer_norm(pair)
        for layer in self.layers:
            x, pair = layer(
                x,
                pair,
                atom_mask=atom_mask,
                pair_mask=pair_mask,
                self_attn_mask=attn_mask,
            )
        return x, pair
