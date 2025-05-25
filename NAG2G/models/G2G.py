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

if __version__ == "2.5.0":
    from unimol.models.unimol import UniMolModel as Unimolv2Model
    from unimol.models.unimol import base_architecture as encoder_base_architecture
from .NAG2G import (
    NAG2GFModel,
    NAG2GFBaseModel,
    decoder_base_architecture,
    base_architecture_fairseq,
    flag_decoder_fairseq,
)
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
        elif __version__ in ["2.0.0", "2.5.0"]:
            Unimolv2Model.add_args(parser)

        if flag_decoder_fairseq:
            NAG2GFBaseModel.default_fairseq_add_args(parser)
        else:
            NAG2GFBaseModel.default_decoder_add_args_main(parser)
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
                "unimolplus",
            ],
            help="model chosen as encoder",
        )

    def __init__(self, args, dictionary, **kwargs):
        NAG2G_G2G_architecture(args)
        super().__init__(args, dictionary, flag_use_base_architecture=False, **kwargs)
        # self.degree_pe = nn.Embedding(12 + 1, self.args.decoder_attention_heads)
        if self.args.decoder_type == "default":
            self.degree_pe = nn.Embedding(100, self.args.decoder_attention_heads)
        elif self.args.decoder_type == "new" or self.args.decoder_type == "fairseq":
            self.degree_pe = nn.Embedding(
                100, self.args.decoder_attention_heads * self.args.reduced_head_dim
            )
            if self.args.want_bond_attn:
                self.bond_pe = nn.Embedding(
                    100, self.args.decoder_attention_heads * self.args.reduced_head_dim
                )
        else:
            raise
        if self.args.want_decoder_attn and self.args.laplacian_pe_dim > 0:
            assert self.args.decoder_type == "default"
            if self.args.not_sumto2:
                self.laplacian_linear = nn.Sequential(
                    nn.Linear(
                        2 * self.args.laplacian_pe_dim, self.args.laplacian_pe_dim
                    ),
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
                multi_gap=self.args.multi_gap,
            )

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
        # laplacian_attn_mask = laplacian_attn_mask.view(
        #     laplacian_attn_mask.shape[0],
        #     laplacian_attn_mask.shape[1],
        #     laplacian_attn_mask.shape[2],
        #     2,
        #     -1,
        # ).permute(0, 1, 2, 4, 3)
        laplacian_attn_mask = self.laplacian_linear(laplacian_attn_mask)
        # laplacian_attn_mask = laplacian_attn_mask.sum(3)
        return laplacian_attn_mask

    def get_encoder(self, kwargs):
        if self.args.encoder_type == "transformerm":
            encoder = self.get_transformerm_encoder(kwargs)
        elif self.args.encoder_type in ["unimolv2", "unimolplus"]:
            encoder = self.get_unimolv2_encoder(kwargs)
            if self.args.weight_encoder != "":
                from unicore import checkpoint_utils

                state = checkpoint_utils.load_checkpoint_to_cpu(
                    self.args.weight_encoder
                )
                # state["model"]["enocder."]
                encoder.load_state_dict(state["model"], strict=False)
        else:
            encoder = super().get_encoder(kwargs)
        return encoder

    def get_attn_mask(self, **kwargs):
        degree_attn_mask = kwargs.pop("decoder_degree_attn_mask", None)
        laplacian_attn_mask = kwargs.pop("decoder_laplacian_attn_mask", None)

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
        degree_attn_mask = kwargs.pop("decoder_degree_attn_mask", None)
        laplacian_attn_mask = kwargs.pop("decoder_laplacian_attn_mask", None)
        bond_attn_mask = kwargs.pop("decoder_bond_attn_mask", None)
        assert degree_attn_mask.max() < 100
        added_degree_attn_mask = self.degree_pe(degree_attn_mask)
        added_degree_attn_mask[degree_attn_mask == 0] = 0
        if self.args.want_bond_attn:
            assert bond_attn_mask.max() < 100
            added_bond_attn_mask = self.bond_pe(bond_attn_mask)
            added_bond_attn_mask[bond_attn_mask == 0] = 0
            added_degree_attn_mask = added_degree_attn_mask + added_bond_attn_mask
        return added_degree_attn_mask

    def get_pad_mask(self, kwargs):
        if self.args.encoder_type == "unimolv2":
            padding_mask = kwargs["batched_data"]["atom_mask"] == 0
            n_mol = padding_mask.shape[0]
        elif self.args.encoder_type == "unimolplus":
            padding_mask = kwargs["batched_data"]["src_token"] == 0
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
        if self.args.encoder_type in ["transformerm", "unimolv2", "unimolplus"]:
            if self.args.add_len == 0:
                padding_mask = self.get_pad_mask(kwargs)
                masked_tokens = ~padding_mask
            else:
                masked_tokens = None
                padding_mask = None
            if self.args.use_reorder and not self.args.encoder_pe_after:
                if self.args.encoder_type == "unimolv2":
                    kwargs["perturb"] = self.embed_positions.weight[
                        : kwargs["batched_data"]["atom_mask"].shape[1], :
                    ].to(kwargs["batched_data"]["atom_mask"].device)
                elif self.args.encoder_type == "unimolplus":
                    kwargs["perturb"] = self.embed_positions.weight[
                        : kwargs["batched_data"]["src_token"].shape[1], :
                    ].to(kwargs["batched_data"]["src_token"].device)
                    kwargs["features_only"] = True
                elif self.args.encoder_type == "transformerm":
                    kwargs["perturb"] = self.embed_positions.weight[
                        : kwargs["batched_data"]["x"].shape[1], :
                    ].to(kwargs["batched_data"]["x"].device)
            output = self.encoder(**kwargs)

            if self.args.encoder_type == "transformerm":
                encoder_rep = output[2]["inner_states"][-1].transpose(0, 1)
            elif self.args.encoder_type == "unimolv2":
                _, _, _, _, encoder_rep = output
            elif self.args.encoder_type == "unimolplus":
                encoder_rep, _, _, _ = output
            if self.args.use_reorder and self.args.encoder_pe_after:
                encoder_rep = encoder_rep + self.embed_positions.weight[
                    : encoder_rep.shape[1], :
                ].to(encoder_rep.device)
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
            elif self.args.decoder_type == "new" or self.args.decoder_type == "fairseq":
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
        elif self.args.encoder_type in ["unimolv2", "unimolplus"]:
            src_tokens = sample["net_input"]["batched_data"]["atom_feat"]

        return src_tokens


@register_model_architecture("G2G", "NAG2G_G2G")
def NAG2G_G2G_architecture(args):
    if __version__ == "1.5.0":
        assert args.encoder_type == "transformerm"
    elif __version__ == "2.0.0":
        assert args.encoder_type == "unimolv2"
    elif __version__ == "2.5.0":
        assert args.encoder_type == "unimolplus"
    if flag_decoder_fairseq:
        base_architecture_fairseq(args)
    else:
        decoder_base_architecture(args)
    encoder_base_architecture(args)
    if __version__ in ["2.0.0", "2.5.0"]:
        assert args.add_len == 0
