import logging

import torch
import torch.nn as nn
from unicore.models import register_model, register_model_architecture

from unimol import __version__

if __version__ == "1.5.0":
    from unimol.models.transformer_m import TransformerMModel
    from unimol.models.transformer_m import (
        bert_base_architecture as encoder_base_architecture,
    )
if __version__ == "2.0.0":
    from unimol.models.unimolv2 import Unimolv2Model
    from unimol.models.unimolv2 import base_architecture as encoder_base_architecture

from .NAG2G import NAG2GFModel, NAG2GFBaseModel, decoder_base_architecture
from unicore import utils
from NAG2G.modules import seq2attn

logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@register_model("G2G")
class G2GModel(NAG2GFModel):
    @staticmethod
    def add_args(parser):
        if __version__ == "1.5.0":
            TransformerMModel.add_args(parser)
            parser.add_argument(
                "--N_vnode",
                type=int,
                default=1,
                metavar="N",
                help="number of vnode",
            )
        elif __version__ == "2.0.0":
            Unimolv2Model.add_args(parser)

        NAG2GFBaseModel.default_decoder_add_args(parser)
        parser.add_argument(
            "--encoder-type",
            default="transformerm",
            choices=[
                "default",
                "transformerm",
                "unimol",
                "default_and_unimol",
                "unimolv2",
            ],
            help="model chosen as encoder",
        )

    def __init__(self, args, dictionary, **kwargs):
        NAG2G_G2G_architecture(args)
        super().__init__(args, dictionary, flag_use_base_architecture=False, **kwargs)
        # self.degree_pe = nn.Embedding(12 + 1, self.args.decoder_attention_heads)
        if self.args.decoder_type == "default":
            self.degree_pe = nn.Embedding(100, self.args.decoder_attention_heads)
        elif self.args.decoder_type == "new":
            self.degree_pe = nn.Embedding(100, self.args.decoder_attention_heads * self.args.reduced_head_dim)
        else:
            raise
        if self.args.want_decoder_attn and self.args.laplacian_pe_dim > 0:
            assert self.args.decoder_type == "default"
            if self.args.not_sumto2:
                self.laplacian_linear = nn.Sequential(
                    nn.Linear(2 * self.args.laplacian_pe_dim, self.args.laplacian_pe_dim),
                    nn.ReLU(),
                    nn.Linear(
                        self.args.laplacian_pe_dim, self.args.decoder_attention_heads
                    ),
                )
            else:
                self.laplacian_linear = nn.Linear(2, self.args.decoder_attention_heads)
        if self.args.want_decoder_attn:
            self.seq2attn = seq2attn(
                self.args.laplacian_pe_dim,
                not self.args.not_sumto2,
                dictionary=dictionary,
                want_h_degree=self.args.want_h_degree,
                idx_type=self.args.idx_type,
                use_class=self.args.use_class,
            )

    def half(self):
        super().half()
        self.encoder = self.encoder.half()
        return self

    def half(self):
        super().half()
        self.encoder = self.encoder.half()
        return self

    def bfloat16(self):
        super().bfloat16()
        self.encoder = self.encoder.bfloat16()
        return self

    def float(self):
        super().float()
        self.encoder = self.encoder.float()
        return self

    def get_transformerm_encoder(self, kwargs):
        encoder = TransformerMModel.build_model(self.args, None)
        return encoder

    def get_unimolv2_encoder(self, kwargs):
        encoder = Unimolv2Model.build_model(self.args, None)
        return encoder

    def get_laplacian_attn_mask(self, laplacian_attn_mask):
        laplacian_attn_mask = self.laplacian_linear(laplacian_attn_mask)
        return laplacian_attn_mask

    def get_encoder(self, kwargs):
        if self.args.encoder_type == "transformerm":
            encoder = self.get_transformerm_encoder(kwargs)
        elif self.args.encoder_type == "unimolv2":
            encoder = self.get_unimolv2_encoder(kwargs)
        else:
            encoder = super().get_encoder(kwargs)
        return encoder

    def get_attn_mask(self, **kwargs):
        degree_attn_mask = (
            kwargs.pop("decoder_degree_attn_mask")
            if "decoder_degree_attn_mask" in kwargs
            else None
        )
        laplacian_attn_mask = (
            kwargs.pop("decoder_laplacian_attn_mask")
            if "decoder_laplacian_attn_mask" in kwargs
            else None
        )

        attn_mask = None
        if degree_attn_mask is not None:
            added_degree_attn_mask = self.degree_pe(degree_attn_mask)
            added_degree_attn_mask[degree_attn_mask == 0] = 0
            added_degree_attn_mask = added_degree_attn_mask.permute(0, 3, 1, 2)
            added_degree_attn_mask = added_degree_attn_mask.reshape(
                -1, degree_attn_mask.shape[1], degree_attn_mask.shape[2]
            )
            if attn_mask is None:
                attn_mask = added_degree_attn_mask
            else:
                attn_mask = attn_mask + added_degree_attn_mask
        if laplacian_attn_mask is not None:
            laplacian_attn_mask = laplacian_attn_mask.to(attn_mask.dtype)
            added_laplacian_attn_mask = self.get_laplacian_attn_mask(
                laplacian_attn_mask
            )
            added_laplacian_attn_mask = added_laplacian_attn_mask.permute(
                0, 3, 1, 2
            ).reshape(-1, laplacian_attn_mask.shape[1], laplacian_attn_mask.shape[2])

            if attn_mask is None:
                attn_mask = added_laplacian_attn_mask
            else:
                attn_mask = attn_mask + added_laplacian_attn_mask
        return attn_mask

    def get_degree_attn_mask(self, **kwargs):
        assert "decoder_degree_attn_mask" in kwargs
        degree_attn_mask = kwargs["decoder_degree_attn_mask"]
        added_degree_attn_mask = self.degree_pe(degree_attn_mask)
        added_degree_attn_mask[degree_attn_mask == 0] = 0
        return added_degree_attn_mask

    def get_pad_mask(self, kwargs):
        if self.args.encoder_type == "unimolv2":
            padding_mask = kwargs["batched_data"]["atom_mask"] == 0
            n_mol = padding_mask.shape[0]
        elif self.args.encoder_type == "transformerm":
            data_x = kwargs["batched_data"]["x"]
            n_mol, n_atom = data_x.size()[:2]
            padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        else:
            raise
        padding_mask_cls = torch.zeros(
            n_mol,
            self.args.N_vnode,
            device=padding_mask.device,
            dtype=padding_mask.dtype,
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        return padding_mask

    def forward_encoder(self, **kwargs):
        if (
            self.args.encoder_type == "transformerm"
            or self.args.encoder_type == "unimolv2"
        ):
            if self.args.add_len == 0:
                padding_mask = self.get_pad_mask(kwargs)
                masked_tokens = ~padding_mask
            else:
                masked_tokens = None
                padding_mask = None
            if self.args.use_reorder:
                if self.args.encoder_type == "unimolv2":
                    kwargs["perturb"] = self.embed_positions.weight[
                        : kwargs["batched_data"]["atom_mask"].shape[1], :
                    ]
                elif self.args.encoder_type == "transformerm":
                    kwargs["perturb"] = self.embed_positions.weight[
                        : kwargs["batched_data"]["x"].shape[1], :
                    ]
            output = self.encoder(**kwargs)

            if self.args.encoder_type == "transformerm":
                encoder_rep = output[2]["inner_states"][-1].transpose(0, 1)
            else:
                _, _, _, _, encoder_rep = output

            return {
                "encoder_rep": encoder_rep,
                "padding_mask": padding_mask,
                "masked_tokens": masked_tokens,
            }
        else:
            return super().forward_encoder(**kwargs)

    def reorder_encoder_out(self, encoder_out, new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_rep"]) == 0:
            new_encoder_rep = None
        else:
            new_encoder_rep = encoder_out["encoder_rep"].index_select(0, new_order)

        if len(encoder_out["padding_mask"]) == 0:
            new_padding_mask = None
        else:
            new_padding_mask = encoder_out["padding_mask"].index_select(0, new_order)

        if len(encoder_out["masked_tokens"]) == 0:
            new_masked_tokens = None
        else:
            new_masked_tokens = encoder_out["masked_tokens"].index_select(0, new_order)

        return {
            "encoder_rep": new_encoder_rep,  # T x B x C
            "padding_mask": new_padding_mask,  # B x T x 1
            "masked_tokens": new_masked_tokens,  # B x T x 1
        }

    def forward_decoder(
        self,
        decoder_src_tokens,
        encoder_cls,
        temperature,
        encoder_padding_mask,
        want_probs=True,
        **kwargs
    ):
        decoder_attn_mask = None
        if self.args.want_decoder_attn:
            if self.args.decoder_attn_from_loader:
                attn_mask_kwargs = kwargs
            elif self.training or not self.args.infer_step:
                attn_mask_kwargs = self.seq2attn.forward_train(decoder_src_tokens)
            else:
                attn_mask_kwargs = self.seq2attn.forward(decoder_src_tokens)
            if self.args.decoder_type == "default":
                decoder_attn_mask = self.get_attn_mask(**attn_mask_kwargs)
            elif self.args.decoder_type == "new":
                decoder_attn_mask = self.get_degree_attn_mask(**attn_mask_kwargs)

        return super().forward_decoder(
            decoder_src_tokens,
            encoder_cls,
            temperature,
            encoder_padding_mask,
            want_probs=want_probs,
            decoder_attn_mask=decoder_attn_mask,
            **kwargs
        )

    def get_src_tokens(self, sample):
        if self.args.encoder_type == "transformerm":
            src_tokens = sample["net_input"]["batched_data"]["x"]
        elif self.args.encoder_type == "unimolv2":
            src_tokens = sample["net_input"]["batched_data"]["atom_feat"]

        return src_tokens


@register_model_architecture("G2G", "NAG2G_G2G")
def NAG2G_G2G_architecture(args):
    if __version__ == "1.5.0":
        assert args.encoder_type == "transformerm"
    elif __version__ == "2.0.0":
        assert args.encoder_type == "unimolv2"

    encoder_base_architecture(args)
    decoder_base_architecture(args)
    if __version__ == "2.0.0":
        assert args.add_len == 0
