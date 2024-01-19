import logging

import numpy as np
import torch
import torch.nn as nn
from unicore import utils
from unimol.data import numpy_seed
from unicore.models import (
    BaseUnicoreModel,
    register_model,
    register_model_architecture,
)
from unicore.modules import (
    LayerNorm,
)

from .layers import (
    AtomFeature,
    EdgeFeature,
    SE3InvariantKernel,
    MovementPredictionHead,
    EnergyHead,
    PredictedLDDTHead,
    Linear,
    Embedding,
)
from .unimolv2_encoder import UniMolv2Encoder

logger = logging.getLogger(__name__)


def init_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear) or isinstance(module, Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding) or isinstance(module, Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


@register_model("unimolv2")
class Unimolv2Model(BaseUnicoreModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--num-3d-bias-kernel",
            type=int,
            default=128,
            metavar="D",
            help="number of kernel in 3D attention bias",
        )
        parser.add_argument(
            "--droppath-prob",
            type=float,
            metavar="D",
            help="stochastic path probability",
            default=0.0,
        )

        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--num-block",
            type=int,
            metavar="N",
            help="number of recycle",
        )
        parser.add_argument(
            "--noise-scale",
            default=0.2,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--label-prob",
            default=0.4,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-prob",
            default=0.2,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-upper",
            default=0.6,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-lower",
            default=0.4,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--plddt-loss-weight",
            default=0.01,
            type=float,
            help="loss weight for plddt",
        )
        parser.add_argument(
            "--pos-loss-weight",
            default=0.2,
            type=float,
            help="loss weight for pos",
        )
        parser.add_argument(
            "--pos-step-size",
            type=float,
            help="step size for pos update",
        )
        parser.add_argument(
            "--gaussian-std-width",
            type=float,
        )
        parser.add_argument(
            "--gaussian-mean-start",
            type=float,
        )
        parser.add_argument(
            "--gaussian-mean-stop",
            type=float,
        )
        parser.add_argument(
            "--pretrain", action="store_true", help="3d pretrain or not"
        )
        parser.add_argument(
            "--N_vnode",
            type=int,
            default=1,
            metavar="N",
            help="number of vnode",
        )

    def __init__(self, args):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.molecule_encoder = UniMolv2Encoder(
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            pair_dim=args.pair_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            activation_fn=args.activation_fn,
            droppath_prob=args.droppath_prob,
        )
        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512
        embedding_dim = args.encoder_embed_dim
        num_attention_heads = args.encoder_attention_heads
        num_3d_bias_kernel = args.num_3d_bias_kernel
        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=embedding_dim,
            N_vnode=args.N_vnode
        )

        self.edge_feature = EdgeFeature(
            pair_dim=args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
            N_vnode=args.N_vnode
        )

        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=num_3d_bias_kernel,
            std_width=args.gaussian_std_width,
            start=args.gaussian_mean_start,
            stop=args.gaussian_mean_stop,
        )
        if not self.args.pretrain:
            self.energy_head = EnergyHead(args.encoder_embed_dim, 1)
        else:
            self.energy_head = None
        self.movement_pred_head = MovementPredictionHead(
            args.encoder_embed_dim, args.pair_embed_dim, args.encoder_attention_heads
        )
        self.lddt_head = PredictedLDDTHead(
            50, args.encoder_embed_dim, args.encoder_embed_dim // 2
        )
        self.movement_pred_head.zero_init()
        self._num_updates = 0
        self.dtype = torch.float32

    def half(self):
        super().half()
        self.se3_invariant_kernel = self.se3_invariant_kernel.float()
        self.atom_feature = self.atom_feature.float()
        self.edge_feature = self.edge_feature.float()
        if self.energy_head is not None:
            self.energy_head = self.energy_head.float()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        self.se3_invariant_kernel = self.se3_invariant_kernel.float()
        self.atom_feature = self.atom_feature.float()
        self.edge_feature = self.edge_feature.float()
        if self.energy_head is not None:
            self.energy_head = self.energy_head.float()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        super().float()
        self.dtype = torch.float32
        return self

    def forward(self, batched_data, perturb=None, cls_embedding=None):
        data_x = batched_data["atom_feat"]
        atom_mask = batched_data["atom_mask"]
        pair_type = batched_data["pair_type"]
        pos = batched_data["pos"]

        if self.args.pretrain:
            with numpy_seed(self.args.seed, self._num_updates, key="recycle"):
                if self.training:
                    num_block = np.random.randint(1, self.args.num_block + 1)
                else:
                    num_block = self.args.num_block
        else:
            num_block = self.args.num_block

        n_mol, n_atom = data_x.shape[:2]
        x = self.atom_feature(batched_data)

        dtype = self.dtype

        x = x.type(dtype)

        if cls_embedding is not None:
            x[:, 0, :] = cls_embedding

        if perturb is not None:
            x[:, self.args.N_vnode:, :] += perturb.type(dtype)

        attn_mask = batched_data["attn_bias"].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.args.encoder_attention_heads, 1, 1
        )
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        attn_bias = self.edge_feature(batched_data, attn_bias)
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat(
            [
                torch.ones(n_mol, self.args.N_vnode, device=atom_mask.device, dtype=atom_mask.dtype),
                atom_mask,
            ],
            dim=1,
        ).type(self.dtype)

        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        def one_block(x, pos, return_x=False):
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)
            attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
            new_attn_bias = attn_bias.clone()
            new_attn_bias[:, self.args.N_vnode:, self.args.N_vnode:, :] = new_attn_bias[:, self.args.N_vnode:, self.args.N_vnode:, :] + attn_bias_3d
            new_attn_bias = new_attn_bias.type(dtype)
            x, pair = self.molecule_encoder(
                x,
                new_attn_bias,
                atom_mask=atom_mask_cls,
                pair_mask=pair_mask,
                attn_mask=attn_mask,
            )
            node_output = self.movement_pred_head(
                x[:, self.args.N_vnode:, :],
                pair[:, self.args.N_vnode:, self.args.N_vnode:, :],
                attn_mask[:, :, self.args.N_vnode:, self.args.N_vnode:],
                delta_pos.detach(),
            )
            node_output = node_output * self.args.pos_step_size
            if return_x:
                return x, pos + node_output
            else:
                return pos + node_output

        if self.args.pretrain:
            with torch.no_grad():
                for _ in range(num_block - 1):
                    pos = one_block(x, pos)
            pos = one_block(x, pos)
            pred_y = None
        else:
            for _ in range(num_block - 1):
                pos = one_block(x, pos)
            x, pos = one_block(x, pos, return_x=True)
            pred_y = self.energy_head(x[:, 0, :]).view(-1)

        pred_dist = (pos.unsqueeze(1) - pos.unsqueeze(2)).norm(dim=-1)

        plddt = self.lddt_head(x[:, self.args.N_vnode:, :])
        return (
            pred_y,
            pos,
            pred_dist,
            plddt,
            x
        )

    @classmethod
    def build_model(cls, args, task):
        return cls(args)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates


@register_model_architecture("unimolv2", "unimolv2_base")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.pair_embed_dim = getattr(args, "pair_embed_dim", 128)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.droppath_prob = getattr(args, "droppath_prob", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.0)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.num_block = getattr(args, "num_block", 4)
    args.pretrain = getattr(args, "pretrain", False)
    args.pos_step_size = getattr(args, "pos_step_size", 0.01)
    args.gaussian_std_width = getattr(args, "gaussian_std_width", 1.0)
    args.gaussian_mean_start = getattr(args, "gaussian_mean_start", 0.0)
    args.gaussian_mean_stop = getattr(args, "gaussian_mean_stop", 9.0)
    args.N_vnode = getattr(args, "N_vnode", 1)
