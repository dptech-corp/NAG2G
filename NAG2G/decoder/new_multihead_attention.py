from typing import Dict, Optional

import torch
from torch import Tensor, nn
from unicore.modules import softmax_dropout


class NewSelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
        reduced_head_dim=4,
        q_reduced_before=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.reduced_head_dim = reduced_head_dim
        self.q_reduced_before = q_reduced_before
        if self.q_reduced_before:
            self.reduced_head_dim_proj = nn.Linear(embed_dim, self.num_heads * self.reduced_head_dim, bias=bias)
        else:
            self.reduced_head_dim_proj = nn.Linear(self.head_dim, self.reduced_head_dim, bias=bias)

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        k_dynamic_T: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim) 
        )
        if k_dynamic_T is not None:
            if self.q_reduced_before:
                # q_reduced = self.reduced_head_dim_proj(q)
                q_reduced = self.reduced_head_dim_proj(query)
                q_reduced = (
                    q_reduced.view(bsz, tgt_len, self.num_heads, self.reduced_head_dim)
                    .transpose(1, 2)
                    .contiguous()
                    .view(-1, self.reduced_head_dim).unsqueeze(1)
                )
            # [batchsize * h, tgt_len, head_dim] -> [batchsize * h * tgt_len, 1, reduced_head_dim]
            else:
                q_reduced = self.reduced_head_dim_proj(q).view(-1, self.reduced_head_dim).unsqueeze(1)
            attn_weights_dynamic = torch.bmm(q_reduced, k_dynamic_T).view(-1, tgt_len, tgt_len)
        q = q * self.scaling
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if k_dynamic_T is not None:
            attn_weights = attn_weights + attn_weights_dynamic

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, bias=attn_bias,
            )
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, inplace=False,
            )

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn