import logging
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from typing import Callable, Optional, Dict, Tuple, Any, NamedTuple, List
import math
from NAG2G.modules import (
    MaskLMHead,
    ClassificationHead,
)
from unicore.modules import init_bert_params, TransformerEncoder, TransformerDecoder
from NAG2G.decoder import TransformerDecoder as NewTransformerDecoder
logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

class NAG2GFBaseModel(BaseUnicoreModel):
    @staticmethod
    def default_encoder_add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )

    @staticmethod
    def default_decoder_add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="H",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="H",
            help="decoder ffn embedding dimension",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--position-type",
            default="normal",
            choices=["sinusoidal", "relative", "normal"],
            help="noise type in coordinate noise",
        )
        # parser.add_argument(
        #     "--transformer-type",
        #     default="normal",
        #     choices=["simple", "normal"],
        #     help="noise type in coordinate noise",
        # )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--contrastive-global-negative",
            action="store_true",
            help="use contrastive learning or not",
        )
        parser.add_argument(
            "--auto-regressive",
            action="store_true",
            help="use auto regressive generative or not",
        )
        parser.add_argument(
            "--class-embedding", action="store_true", help="use class embedding or not"
        )
        parser.add_argument(
            "--use-decoder", action="store_true", help="use decoder or not"
        )
        parser.add_argument(
            "--smoothl1-beta",
            default=1.0,
            type=float,
            help="beta in pair distance smoothl1 loss",
        )
        parser.add_argument(
            "--rel_pos",
            action="store_true",
            help="rel_pos",
        )
        parser.add_argument(
            "--flag_old",
            action="store_true",
            help="flag_old",
        )
        parser.add_argument(
            "--decoder_type",
            default="default",
            choices=[
                "default",
                "new",
            ],
            help="model chosen as decoder",
        )
        parser.add_argument(
            "--reduced_head_dim", type=int, default=4, help="reduced_head_dim"
        )
        parser.add_argument(
            "--q_reduced_before",
            action="store_true",
            help="q_reduced_before",
        )
        
        parser.add_argument(
            "--want_emb_k_dynamic_proj",
            action="store_true",
            help="want_emb_k_dynamic_proj",
        )

        parser.add_argument(
            "--want_emb_k_dynamic_dropout",
            action="store_true",
            help="want_emb_k_dynamic_dropout",
        )
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        NAG2GFBaseModel.default_encoder_add_args(parser)
        NAG2GFBaseModel.default_decoder_add_args(parser)

    def __init__(self, args, dictionary, **kwargs):
        super().__init__()
        flag_use_base_architecture = (
            kwargs.pop("flag_use_base_architecture")
            if "flag_use_base_architecture" in kwargs
            else True
        )
        if flag_use_base_architecture:
            base_architecture(args)
        self.init(args, dictionary, **kwargs)

    def init(self, args, dictionary, **kwargs):
        self.args = args
        self.padding_idx = dictionary.pad()
        if self.args.bpe_tokenizer_path == "none":
            len_dict = len(dictionary)
        else:
            with open(os.path.join(self.args.bpe_tokenizer_path, "vocab.json"), "r") as f:
                len_dict = len(list(json.load(f).keys()))
        self.embed_tokens = nn.Embedding(
            len_dict, args.encoder_embed_dim, self.padding_idx
        )            
        self.encoder = self.get_encoder(kwargs)

        self.lm_head = MaskLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len_dict,
            activation_fn=args.activation_fn,
            weight=None,
        )
        self.auto_regressive = args.auto_regressive
        self.use_decoder = args.use_decoder
        # self.embed_positions = self.get_position_embedding('test', args.max_seq_len, args.encoder_embed_dim)

        self.embed_positions = self.get_position_embedding(
            args.position_type, args.max_seq_len, args.encoder_embed_dim
        )
        # self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)

        self.use_class_embedding = args.class_embedding
        if self.use_class_embedding:
            self.class_embedding = nn.Embedding(100, args.encoder_embed_dim)

        if args.auto_regressive:
            self.use_decoder = True
        if self.use_decoder:
            # self.decoder_embed_positions = self.get_position_embedding('test', args.max_seq_len, args.decoder_embed_dim)
            # self.decoder_embed_positions = nn.Embedding(args.max_seq_len, args.decoder_embed_dim)
            self.decoder_embed_positions = self.get_position_embedding(
                args.position_type, args.max_seq_len, args.decoder_embed_dim
            )
            self.decoder_embed_tokens = self.embed_tokens  # FFFFFF
            # self.decoder_embed_tokens = nn.Embedding(len(dictionary), args.decoder_embed_dim, self.padding_idx)
            self.decoder = self.get_decoder()
            self.decoder_lm_head = MaskLMHead(
                embed_dim=args.decoder_embed_dim,
                output_dim=len_dict,
                activation_fn=args.activation_fn,
                weight=None,
            )
            # self.decoder_lm_head = nn.Linear(args.decoder_embed_dim, len(dictionary))
        self.classification_heads = nn.ModuleDict()

        self.apply(init_bert_params)
        print("flag_old", self.args.flag_old)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if hasattr(task, "encoder_dictionary"):
            return cls(
                args, task.dictionary, encoder_dictionary=task.encoder_dictionary
            )
        return cls(args, task.dictionary)

    def get_default_encoder(self):
        encoder = TransformerEncoder(
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
        )
        return encoder

    def get_encoder(self, kwargs):
        encoder = self.get_default_encoder()
        return encoder

    def get_decoder(self):
        if self.args.decoder_type == "default":
            decoder = TransformerDecoder(
                decoder_layers=self.args.decoder_layers,
                embed_dim=self.args.decoder_embed_dim,
                ffn_embed_dim=self.args.decoder_ffn_embed_dim,
                attention_heads=self.args.decoder_attention_heads,
                emb_dropout=self.args.emb_dropout,
                dropout=self.args.dropout,
                attention_dropout=self.args.attention_dropout,
                activation_dropout=self.args.activation_dropout,
                max_seq_len=self.args.max_seq_len,
                activation_fn=self.args.activation_fn,
                auto_regressive=self.args.auto_regressive,
                post_ln=self.args.post_ln,
                rel_pos=self.args.rel_pos,
            )
        elif self.args.decoder_type == "new":
            decoder = NewTransformerDecoder(
                decoder_layers=self.args.decoder_layers,
                embed_dim=self.args.decoder_embed_dim,
                ffn_embed_dim=self.args.decoder_ffn_embed_dim,
                attention_heads=self.args.decoder_attention_heads,
                emb_dropout=self.args.emb_dropout,
                dropout=self.args.dropout,
                attention_dropout=self.args.attention_dropout,
                activation_dropout=self.args.activation_dropout,
                max_seq_len=self.args.max_seq_len,
                activation_fn=self.args.activation_fn,
                auto_regressive=self.args.auto_regressive,
                post_ln=self.args.post_ln,
                rel_pos=self.args.rel_pos,
                reduced_head_dim = self.args.reduced_head_dim,
                q_reduced_before = self.args.q_reduced_before,
                want_emb_k_dynamic_proj = self.args.want_emb_k_dynamic_proj,
                want_emb_k_dynamic_dropout = self.args.want_emb_k_dynamic_dropout,
            )
        else:
            raise
        return decoder

    def get_position_embedding(self, position_type, max_seq_len, embed_dim):

        if position_type == "sinusoidal":
            pe = torch.zeros(max_seq_len, embed_dim)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, embed_dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / embed_dim)
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe1 = nn.Embedding(max_seq_len, embed_dim)
            pe1.weight = nn.Parameter(pe, requires_grad=False)
            return pe1

        elif position_type == "relative":
            # relative_pe = nn.Embedding(max_seq_len * 2 + 2, embed_dim)
            pe = torch.zeros(max_seq_len, embed_dim // 2)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, (embed_dim // 2), 2, dtype=torch.float)
                    * -(math.log(10000.0) / (embed_dim // 2))
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe1 = nn.Embedding(max_seq_len, embed_dim // 2)
            pe1.weight = nn.Parameter(pe, requires_grad=False)
            relative = nn.Embedding(max_seq_len, embed_dim // 2)
            relative_pe = torch.cat((relative, pe1), -1)
            return relative_pe
        else:
            return nn.Embedding(max_seq_len, embed_dim)

    def forward(
        self,
        # src_tokens,
        # decoder_src_tokens,
        # relation_type,
        # masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        decoder_kwargs = {}
        encoder_kwargs = {}
        for k, v in kwargs.items():
            if "decoder" in k:
                decoder_kwargs[k] = v
            else:
                encoder_kwargs[k] = v
        if self.args.use_class_encoder:
            assert self.args.N_vnode == 2
            encoder_kwargs["cls_embedding"] = self.decoder_embed_tokens(decoder_kwargs["decoder_src_tokens"][:, 1])
        encoder_result = self.forward_encoder(
            # src_tokens=src_tokens,
            **encoder_kwargs
        )
        masked_tokens = encoder_result.pop("masked_tokens")
        encoder_rep = encoder_result.pop("encoder_rep")
        padding_mask = encoder_result.pop("padding_mask")

        decoder_outprob, vae_kl_loss = self.forward_decoder(
            encoder_cls=encoder_rep,
            temperature=None,
            encoder_padding_mask=padding_mask,
            want_probs=False,
            **decoder_kwargs,
        )

        contrast_out = None
        if not features_only:
            logits = self.lm_head(encoder_rep, masked_tokens)
        else:
            logits = encoder_rep

        return logits, decoder_outprob, contrast_out, vae_kl_loss

    def forward_default_encoder(self, encoder, src_tokens, **kwargs):
        padding_mask = src_tokens.eq(self.padding_idx)
        masked_tokens = ~padding_mask
        tmp_padding_mask = padding_mask
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        seq_len = src_tokens.size(1)
        x = x * math.sqrt(x.shape[-1])  # FFFFFF
        x += self.embed_positions.weight[:seq_len, :]
        # if self.use_class_embedding:
        #     x[:,0,:] += self.class_embedding(relation_type)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        encoder_rep = encoder(x, padding_mask=padding_mask, attn_mask=None)
        return encoder_rep, padding_mask, tmp_padding_mask, masked_tokens

    def forward_encoder(
        self,
        src_tokens,
        # relation_type,
        # masked_tokens=None,
        **kwargs
    ):
        encoder_rep, padding_mask, _, masked_tokens = self.forward_default_encoder(
            self.encoder, src_tokens
        )
        return {
            "encoder_rep": encoder_rep,
            "padding_mask": padding_mask,
            "masked_tokens": padding_mask,
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
        decoder_outprob = None
        vae_kl_loss = None
        if self.use_decoder:
            decoder_padding_mask = decoder_src_tokens.eq(self.padding_idx)
            if not decoder_padding_mask.any():
                decoder_padding_mask = None
            x_decoder = self.decoder_embed_tokens(decoder_src_tokens)
            if self.args.flag_old:
                x_decoder = x_decoder * math.sqrt(x_decoder.shape[-1])  # FFFFFF
            seq_len = decoder_src_tokens.size(1)
            x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
            if self.args.flag_old and decoder_padding_mask is not None:
                x_decoder = x_decoder * (
                    1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder)
                )
            attn_mask = (
                kwargs["decoder_attn_mask"]
                if "decoder_attn_mask" in kwargs.keys()
                else None
            )
            if self.args.decoder_type == "default":
                new_dict = {"attn_mask": attn_mask}
            elif self.args.decoder_type == "new":
                new_dict = {"emb_k_dynamic": attn_mask}
            decoder_rep = self.decoder(
                x_decoder,
                padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
                encoder_out=encoder_cls,
                **new_dict,
            )

            decoder_outprob = self.decoder_lm_head(decoder_rep)
            if want_probs:
                probs = self.get_normalized_probs(
                    decoder_outprob, temperature, log_probs=True, sample=None
                )
                probs = probs[:, -1, :]
                return probs, None

        return decoder_outprob, vae_kl_loss

    def get_normalized_probs(self, net_output, temperature, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output  # [0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return torch.log(F.softmax(logits / temperature, dim=-1))

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )


@register_model("NAG2GF")
class NAG2GFModel(NAG2GFBaseModel):
    @staticmethod
    def add_args(parser):
        NAG2GFBaseModel.add_args(parser)
        parser.add_argument(
            "--encoder-type",
            default="default",
            choices=["default", "unimol", "default_and_unimol"],
            help="model chosen as encoder",
        )

    def get_unimol_encoder(self, kwargs):
        encoder_dictionary = (
            kwargs["encoder_dictionary"]
            if "encoder_dictionary" in kwargs.keys()
            else None
        )
        assert encoder_dictionary is not None
        from .unimol_encoder import CustomizedUniMolModel

        encoder = CustomizedUniMolModel(self.args, encoder_dictionary)
        return encoder

    def get_encoder(self, kwargs):
        if self.args.encoder_type == "default":
            encoder = self.get_default_encoder()
        elif self.args.encoder_type == "unimol":
            encoder = self.get_unimol_encoder(kwargs)
        elif self.args.encoder_type == "default_and_unimol":
            encoder = nn.ModuleList(
                [self.get_default_encoder(), self.get_unimol_encoder(kwargs)]
            )
        return encoder

    def forward_unimol_encoder(self, encoder, src_tokens, **kwargs):
        padding_mask = src_tokens.eq(self.padding_idx)
        masked_tokens = ~padding_mask
        if not padding_mask.any():
            padding_mask = None
        tmp_padding_mask = padding_mask
        encoder_input_dict = {
            "src_tokens": src_tokens,
            "encoder_masked_tokens": masked_tokens,
            "src_distance": kwargs["src_distance"],
            "src_coord": kwargs["src_coord"],
            "src_edge_type": kwargs["src_edge_type"],
            "features_only": True,
        }

        encoder_rep, _, _, _, _, = encoder(
            **encoder_input_dict,
        )
        return encoder_rep, padding_mask, tmp_padding_mask, masked_tokens

    def forward_encoder(
        self,
        src_tokens,
        # relation_type,
        # masked_tokens=None,
        **kwargs
    ):
        if self.args.encoder_type == "default":
            encoder_rep, padding_mask, _, masked_tokens = self.forward_default_encoder(
                self.encoder, src_tokens
            )

        elif self.args.encoder_type == "unimol":
            encoder_rep, padding_mask, _, masked_tokens = self.forward_unimol_encoder(
                self.encoder, src_tokens, **kwargs
            )

        elif self.args.encoder_type == "default_and_unimol":
            (
                default_encoder_rep,
                default_padding_mask,
                default_padding_mask_tmp,
                default_masked_tokens,
            ) = self.forward_default_encoder(
                self.encoder[0], src_tokens=kwargs["smiles_src_tokens"]
            )
            (
                unimol_encoder_rep,
                unimol_padding_mask,
                unimol_padding_mask_tmp,
                unimol_masked_tokens,
            ) = self.forward_unimol_encoder(self.encoder[1], src_tokens, **kwargs)
            encoder_rep = torch.cat([default_encoder_rep, unimol_encoder_rep], 1)
            masked_tokens = torch.cat([default_masked_tokens, unimol_masked_tokens], 1)
            if default_padding_mask is None and unimol_padding_mask is None:
                padding_mask = None
            else:
                padding_mask = torch.cat(
                    [default_padding_mask_tmp, unimol_padding_mask_tmp], 1
                )

        return {
            "encoder_rep": encoder_rep,
            "padding_mask": padding_mask,
            "masked_tokens": masked_tokens,
        }

    def get_src_tokens(self, sample):
        src_tokens = sample["net_input"]["src_tokens"]  # B x T
        return src_tokens


@register_model_architecture("NAG2GF", "NAG2GF")
def base_architecture(args):
    encoder_base_architecture(args)
    decoder_base_architecture(args)


def encoder_base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")


def decoder_base_architecture(args):

    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.post_ln = getattr(args, "post_ln", False)
    args.contrastive_global_negative = getattr(
        args, "contrastive_global_negative", False
    )
    args.auto_regressive = getattr(args, "auto_regressive", False)
    args.use_decoder = getattr(args, "use_decoder", False)
    args.class_embedding = getattr(args, "class_embedding", False)

    args.decoder_layers = getattr(args, "decoder_layers", 15)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    args.decoder_loss = getattr(args, "decoder_loss", 1)
    args.rel_pos = getattr(args, "rel_pos", False)
    args.flag_old = getattr(args, "flag_old", False)
    args.decoder_type = getattr(args, "decoder_type", "default")
    args.reduced_head_dim = getattr(args, "reduced_head_dim", 4)
    args.q_reduced_before = getattr(args, "q_reduced_before", False)
    args.want_emb_k_dynamic_proj = getattr(args, "want_emb_k_dynamic_proj", False)
    args.want_emb_k_dynamic_dropout = getattr(args, "want_emb_k_dynamic_dropout", True)
    # args.encoder_type = getattr(args, "encoder_type", "default")


@register_model_architecture("NAG2GF", "NAG2GF_base")
def NAG2G_base_architecture(args):
    base_architecture(args)


@register_model_architecture("NAG2GF", "NAG2GF_unimol")
def NAG2G_unimol_architecture(args):
    args.encoder_type = "unimol"
    base_architecture(args)


@register_model_architecture("NAG2GF", "NAG2GF_DnU")
def NAG2G_default_and_unimol_architecture(args):
    args.encoder_type = "default_and_unimol"
    base_architecture(args)
