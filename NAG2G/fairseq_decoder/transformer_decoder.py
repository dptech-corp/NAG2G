import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from unicore.modules import LayerNorm
from unicore.modules import LayerNorm
from typing import Any, Dict, List, Tuple, Optional
from NAG2G.fairseq_modules.transformer_decoder import (
    TransformerDecoderBase as Old_TransformerDecoderBase,
)
from NAG2G.fairseq_decoder.transformer_layer import TransformerDecoderLayerBase
from NAG2G.utils.utils import checkpoint_wrapper
from unicore.modules.transformer_encoder import relative_position_bucket


class TransformerDecoderBase(Old_TransformerDecoderBase):
    def __init__(self, *args, **kwargs):
        embed_positions = kwargs.pop("embed_positions", None)
        super().__init__(*args, **kwargs)
        self.embed_positions = embed_positions
        self.init_emb_k_dynamic()
        self.init_rel_pos()

    def init_emb_k_dynamic(self):
        embed_dim = self.cfg.decoder.attention_heads * self.cfg.reduced_head_dim
        self.embed_k_scale = (
            1.0 if self.cfg.no_scale_embedding else math.sqrt(embed_dim)
        )
        self.k_dynamic_scaling = self.cfg.reduced_head_dim**-0.5
        self.emb_k_dynamic_layer_norm = LayerNorm(embed_dim)
        if self.cfg.want_emb_k_dynamic_proj:
            self.emb_k_dynamic_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            nn.init.normal_(
                self.emb_k_dynamic_proj.weight, mean=0, std=embed_dim**-0.5
            )

    def init_rel_pos(self):
        if self.cfg.rel_pos:
            assert self.cfg.rel_pos_bins % 2 == 0
            self.relative_attention_bias = nn.Embedding(
                self.cfg.rel_pos_bins, self.cfg.decoder.attention_heads
            )
            seq_len = self.cfg.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.cfg.rel_pos_bins,
                max_distance=self.cfg.max_rel_pos,
            )
            self.rp_bucket -= self.rp_bucket.min()

    def get_rel_pos_bias(self, device, seq_len, incremental_state):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != device:
            self.rp_bucket = self.rp_bucket.to(device)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        if incremental_state is not None:
            rp_bucket = rp_bucket[-1:, :]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def get_emb_k_dynamic_T(self, emb_k_dynamic, seq_len, incremental_state):
        if emb_k_dynamic is not None:
            if incremental_state is not None:
                emb_k_dynamic = emb_k_dynamic[:, -1:, :, :]
                seq_len_1 = 1
            else:
                seq_len_1 = seq_len
            bsz = emb_k_dynamic.shape[0]
            emb_k_dynamic = self.embed_k_scale * emb_k_dynamic
            emb_k_dynamic = self.emb_k_dynamic_layer_norm(emb_k_dynamic)
            if self.cfg.want_emb_k_dynamic_dropout:
                emb_k_dynamic = self.dropout_module(emb_k_dynamic)
            if self.cfg.want_emb_k_dynamic_proj:
                emb_k_dynamic = self.emb_k_dynamic_proj(emb_k_dynamic)
            # [batchsize, n, n, head_dim * reduced_head_dim] -> [batchsize * h * seq_len_1, seq_len, reduced_head_dim]
            emb_k_dynamic = (
                emb_k_dynamic.view(
                    bsz,
                    seq_len_1,
                    seq_len,
                    self.cfg.decoder.attention_heads,
                    self.cfg.reduced_head_dim,
                )
                .permute(0, 3, 1, 2, 4)
                .contiguous()
                .view(
                    bsz * self.cfg.decoder.attention_heads * seq_len_1,
                    seq_len,
                    self.cfg.reduced_head_dim,
                )
                .transpose(1, 2)
            ) * self.k_dynamic_scaling
        return emb_k_dynamic

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        if encoder_out is not None and len(encoder_out["emb_k_dynamic"]) > 0:
            emb_k_dynamic_T = self.get_emb_k_dynamic_T(
                encoder_out["emb_k_dynamic"][0], slen, incremental_state
            )

        # embed positions
        positions = None
        if self.embed_positions is not None:
            #     positions = self.embed_positions(
            #         prev_output_tokens, incremental_state=incremental_state
            #     )
            positions = self.embed_positions.weight[:slen, :]

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[-1:, :]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if self_attn_padding_mask is not None:
            x = x * (1 - self_attn_padding_mask.unsqueeze(-1).type_as(x))

        rel_pos_bias = None
        if self.cfg.rel_pos:
            rel_pos_bias = self.get_rel_pos_bias(x.device, slen, incremental_state)
            rel_pos_bias_shape = rel_pos_bias.shape
            rel_pos_bias = rel_pos_bias.expand(x.size(0), -1, -1, -1).reshape(
                rel_pos_bias_shape[0] * x.size(0),
                rel_pos_bias_shape[1],
                rel_pos_bias_shape[2],
            )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
                if rel_pos_bias is not None:
                    self_attn_mask = self_attn_mask.unsqueeze(0) + rel_pos_bias
            else:
                # self_attn_mask = None
                self_attn_mask = rel_pos_bias
            x, layer_attn, _ = layer(
                x=x,
                k_dynamic_T=emb_k_dynamic_T,
                encoder_out=enc,
                encoder_padding_mask=padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
