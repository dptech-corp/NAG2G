# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import re
import ast
import inspect
from dataclasses import _MISSING_TYPE, dataclass, field, fields, is_dataclass
from dataclasses import MISSING as MISSING_db
from typing import Any, Dict, List, Optional, Tuple, Type
from enum import Enum, EnumMeta
from argparse import ArgumentError, ArgumentParser, Namespace

import torch
from omegaconf import II, MISSING
from unicore import utils
from NAG2G.utils.utils import safe_getattr, safe_hasattr

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

_NAME_PARSER = r"(decoder|encoder|quant_noise)_(.*)"



class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))

class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


@dataclass
class FairseqDataclass:
    """fairseq base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")

    @classmethod
    def from_namespace(cls, args):
        if isinstance(args, cls):
            return args
        else:
            config = cls()
            for k in config.__dataclass_fields__.keys():
                if k.startswith("_"):
                    # private member, skip
                    continue
                if hasattr(args, k):
                    setattr(config, k, getattr(args, k))

            return config

def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]

def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(
        r"(typing.|^)Union\[(.*), NoneType\]$", typestring
    ) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type

def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance: FairseqDataclass,
    delete_default: bool = False,
    with_prefix: Optional[str] = None,
) -> None:
    """
    convert a dataclass instance to tailing parser arguments.
    If `with_prefix` is provided, prefix all the keys in the resulting parser with it. It means that we are
    building a flat namespace from a structured dataclass (see transformer_config.py for example).
    """

    def argparse_name(name: str):
        if name == "data" and (with_prefix is None or with_prefix == ""):
            # normally data is positional args, so we don't add the -- nor the prefix
            return name
        if name == "_name":
            # private member, skip
            return None
        full_name = "--" + name.replace("_", "-")
        if with_prefix is not None and with_prefix != "":
            # if a prefix is specified, construct the prefixed arg name
            full_name = with_prefix + "-" + full_name[2:]  # strip -- when composing
        return full_name

    def get_kwargs_from_dc(
        dataclass_instance: FairseqDataclass, k: str
    ) -> Dict[str, Any]:
        """k: dataclass attributes"""

        kwargs = {}

        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)

        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING_db:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (
                isinstance(inter_type, type)
                and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
            ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError(
                        "parsing of type " + str(inter_type) + " is not implemented"
                    )
                if field_default is not MISSING_db:
                    kwargs["default"] = (
                        ",".join(map(str, field_default))
                        if field_default is not None
                        else None
                    )
            elif (
                isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING_db:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING_db:
                    kwargs["default"] = field_default

        # build the help with the hierarchical prefix
        if with_prefix is not None and with_prefix != "" and field_help is not None:
            field_help = with_prefix[2:] + ": " + field_help

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, FairseqDataclass):
            # for fields that are of type FairseqDataclass, we can recursively
            # add their fields to the namespace (so we add the args from model, task, etc. to the root namespace)
            prefix = None
            if with_prefix is not None:
                # if a prefix is specified, then we don't want to copy the subfields directly to the root namespace
                # but we prefix them with the name of the current field.
                prefix = field_name
            gen_parser_from_dataclass(parser, field_type(), delete_default, prefix)
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                "${"
            ):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default and "default" in kwargs:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass

@dataclass
class EncDecBaseConfig(FairseqDataclass):
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
    )
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for FFN"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each block"}
    )
    learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    layerdrop: float = field(default=0, metadata={"help": "LayerDrop probability"})
    layers_to_keep: Optional[List[int]] = field(
        default=None, metadata={"help": "which layers to *keep* when pruning"}
    )

    xformers_att_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "config for xFormers attention, defined in xformers.components.attention.AttentionConfig"
        },
    )


class DecoderConfig(EncDecBaseConfig):
    input_dim: int = II("model.decoder.embed_dim")
    output_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )

    def __post_init__(self):
        #  II doesn't work if we are just creating the object outside of hydra so fix that
        if self.input_dim == II("model.decoder.embed_dim"):
            self.input_dim = self.embed_dim
        if self.output_dim == II("model.decoder.embed_dim"):
            self.output_dim = self.embed_dim


@dataclass
class QuantNoiseConfig(FairseqDataclass):
    pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )

@dataclass
class TransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu",
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    adaptive_input: bool = False
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    # TODO should really be in the encoder config
    max_source_positions: int = field(
        default=DEFAULT_MAX_SOURCE_POSITIONS,
        metadata={"help": "Maximum input length supported by the encoder"},
    )
    decoder: DecoderConfig = DecoderConfig()
    # TODO should really be in the decoder config
    max_target_positions: int = field(
        default=DEFAULT_MAX_TARGET_POSITIONS,
        metadata={"help": "Maximum output length supported by the decoder"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={
            "help": "share encoder, decoder and output embeddings (requires shared dictionary and embed dim)"
        },
    )
    merge_src_tgt_embed: bool = field(
        default=False,
        metadata={
            "help": "if true then the source and target embedding table is "
            "merged into one table. This is going to make the model smaller but "
            "it might hurt performance."
        },
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    adaptive_softmax_cutoff: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0.0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    no_cross_attention: bool = field(
        default=False, metadata={"help": "do not perform cross-attention"}
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "perform cross+self-attention"}
    )
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )
    # DEPRECATED field, but some old checkpoints might have it
    char_inputs: bool = field(
        default=False, metadata={"help": "if set, model takes character ids as input"}
    )
    relu_dropout: float = 0.0
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1,
        metadata={"help": "shuffle tokens between workers before computing assignment"},
    )

    export: bool = field(
        default=False,
        metadata={"help": "make the layernorm exportable with torchscript."},
    )

    # copied from transformer_lm but expected in transformer_decoder:
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )

    # We need to make this hierarchical dataclass like the flat namespace
    # __getattr__ and __setattr__ here allow backward compatibility
    # for subclasses of Transformer(Legacy) that depend on read/write on
    # the flat namespace.

    def __getattr__(self, name):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"invalid argument {name}.")

    def __setattr__(self, name, value):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _copy_keys(args, cls, prefix, seen):
        """
        copy the prefixed keys (decoder_embed_dim) to the DC fields: decoder.embed_dim
        """
        cfg = cls()
        for fld in fields(cls):
            # for all the fields in the DC, find the fields (e.g. embed_dim)
            # in the namespace with the prefix (e.g. decoder)
            # and set it on the dc.
            args_key = f"{prefix}_{fld.name}"
            if safe_hasattr(args, args_key):
                seen.add(args_key)
                setattr(cfg, fld.name, safe_getattr(args, args_key))
            if safe_hasattr(args, fld.name):
                seen.add(fld.name)
                setattr(cfg, fld.name, safe_getattr(args, fld.name))
        return cfg


    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
