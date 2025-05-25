import torch
import functools
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig
import re
import torch.utils.checkpoint as checkpoint
from unicore import utils
from itertools import accumulate

import logging
logger = logging.getLogger(__name__)

def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]
        
def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def get_perplexity(loss, round=2, base=2):
    from unicore.logging.meters import safe_round

    if loss is None:
        return 0.0
    try:
        return safe_round(base ** loss, round)
    except OverflowError:
        return float("inf")

def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.
    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.
    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding

def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

def safe_getattr(obj, k, default=None):
    """Returns obj[k] if it exists and is not None, otherwise returns default."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)

def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None

def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == "xla":
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

def checkpoint_wrapper(m, offload_to_cpu=False):
    """
    A friendlier wrapper for performing activation checkpointing.
    Compared to the PyTorch version, this version:
    - wraps an nn.Module, so that all subsequent calls will use checkpointing
    - handles keyword arguments in the forward
    - handles non-Tensor outputs from the forward
    Usage::
        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    # should I check whether original_forward has already been set?
    assert not hasattr(
        m, "precheckpoint_forward"
    ), "checkpoint function has already been applied?"
    m.precheckpoint_forward = m.forward
    m.forward = functools.partial(
        _checkpointed_forward,
        m.precheckpoint_forward,  # original_forward
        offload_to_cpu,
    )
    return m

def _checkpointed_forward(original_forward, offload_to_cpu, *args, **kwargs):
    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict = {"offload": offload_to_cpu}
    output = CheckpointFunction.apply(
        original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    if isinstance(output, torch.Tensor):
        return output
    else:
        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
        return output

def pack_kwargs(*args, **kwargs) -> Tuple[List[str], List[Any]]:
    """
    Usage::
        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args

def prune_state_dict(state_dict, model_cfg: Optional[DictConfig]):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).
    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.
    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            int(layer_string) for layer_string in layers_to_keep.split(",")
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if isinstance(model_cfg, DictConfig):
        context = open_dict(model_cfg)
    else:
        context = contextlib.ExitStack()
    with context:
        if hasattr(model_cfg, "encoder_layers_to_keep"):
            model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"):
            model_cfg.decoder_layers_to_keep = None

    return new_state_dict

class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.
    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling ``unpack_non_tensors``.
    """

    @staticmethod
    def forward(ctx, run_function, parent_ctx_dict, kwarg_keys, *args):
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = utils.get_rng_state()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(
                x.to(torch.device("cpu"), non_blocking=True) for x in tensor_inputs
            )

        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs
            return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        tensor_inputs: Tuple = ctx.saved_tensors
        tensor_inputs = checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = [
                t.to(ctx.fwd_device[i], non_blocking=True)
                for i, t in enumerate(tensor_inputs)
            ]
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = utils.get_rng_state()

        # Set the states to what it used to be before the forward pass.
        utils.set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)
        # Set the states back to what it was at the start of this function.
        utils.set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )
        return (None, None, None) + grads

def unpack_kwargs(
    kwarg_keys: List[str], flat_args: List[Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs

def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any]]
) -> Tuple[Tuple[torch.Tensor], Dict[str, List[Any]]]:
    """
    Usage::
        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors = []
    packed_non_tensors = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors

def unpack_non_tensors(
    tensors: Tuple[torch.Tensor],
    packed_non_tensors: Dict[str, List[Any]],
) -> Tuple[Any]:
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)


def get_token_to_word_mapping(tokens, exclude_list):
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = (
        ((tgt_sent != pad) & (tgt_sent != eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    src_invalid = (
        ((src_sent == pad) | (src_sent == eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float("-inf")
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append(
                (
                    src_token_to_word[src_idx.item()] - 1,
                    tgt_token_to_word[tgt_idx.item()] - 1,
                )
            )
    return alignment


def extract_soft_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = ((tgt_sent != pad)).nonzero(as_tuple=False)
    src_valid = ((src_sent != pad)).nonzero(as_tuple=False).squeeze(dim=-1)
    alignment = []
    if len(tgt_valid) != 0 and len(src_valid) != 0:
        attn_valid = attn[tgt_valid, src_valid]
        alignment = [
            ["{:.6f}".format(p) for p in src_probs.tolist()] for src_probs in attn_valid
        ]
    return alignment

def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]
