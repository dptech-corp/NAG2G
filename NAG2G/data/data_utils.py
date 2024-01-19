try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import logging
import numpy as np

logger = logging.getLogger(__name__)


def collate_cross_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size_h - v.size(0):, size_w - v.size(1):]
                    if left_pad else res[i][:v.size(0), :v.size(1)])
    return res


def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, :]
                    if left_pad else res[i][: len(v), :])
    return res


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.
    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def batch_by_dynamic(
    indices,
    num_tokens_fn,
    num_tokens_vec=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.
    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """

    # added int() to avoid TypeError: an integer is required
    max_tokens = int(max_tokens) if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        if num_tokens_vec is None:
            return batch_by_size_fn(
                indices,
                num_tokens_fn,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        else:
            return batch_by_size_vec(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )

    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort(
            [
                fixed_shapes[:, 1].argsort(),  # length
                fixed_shapes[:, 0].argsort(),  # bsz
            ]
        )
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def batch_by_size_vec(
    indices,
    num_tokens_vec,
    max_tokens,
    max_sentences,
    bsz_mult,
):
    if indices.shape[0] == 0:
        return []
    assert max_tokens <= 0 or np.max(num_tokens_vec) <= max_tokens, (
        f"Sentences lengths should not exceed max_tokens={max_tokens}"
    )

    indices_len = indices.shape[0]
    batches_ends = np.zeros(indices_len, dtype=np.int32)
    batches_ends_view = batches_ends
    num_tokens_view = num_tokens_vec

    pos = 0
    new_batch_end = 0

    new_batch_max_tokens = 0
    new_batch_sentences = 0
    new_batch_num_tokens = 0

    overflow = False
    size_matches_with_bsz_mult = False

    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        # At every pos we keep stats about the last complete batch [batch_start:batch_end),
        #      and tail [batch_end:pos].
        # 1) Every time when (batch + tail) forms a valid batch
        #      (according to max_tokens, max_sentences and bsz_mult) we append tail to batch.
        # 2) When (batch+tail) violates max_tokens or max_sentences constraints
        #      we finalize running batch, and tail becomes a new batch.
        # 3) There is a corner case when tail also violates constraints.
        #      In that situation [batch_end:pos-1] (tail without the current pos)
        #      gets added to the finalized batches, while [pos:pos] becomes a new tail.
        #
        # Important: For the sake of performance try to avoid using function calls within this loop.

        tail_max_tokens = tail_max_tokens \
            if tail_max_tokens > num_tokens_view[pos] \
            else num_tokens_view[pos]
        new_batch_end = pos + 1
        new_batch_max_tokens = batch_max_tokens \
            if batch_max_tokens > tail_max_tokens \
            else tail_max_tokens
        new_batch_sentences = new_batch_end - batch_start
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens
        overflow = (new_batch_sentences > max_sentences > 0 or
                    new_batch_num_tokens > max_tokens > 0)
        size_matches_with_bsz_mult = (new_batch_sentences < bsz_mult or
                                      new_batch_sentences % bsz_mult == 0)

        if overflow:
            tail_num_tokens = tail_max_tokens * \
                (new_batch_end - batches_ends_view[batches_count])
            tail_overflow = tail_num_tokens > max_tokens > 0
            # In case of a tail overflow finalize two batches
            if tail_overflow:
                batches_count += 1
                batches_ends_view[batches_count] = pos
                tail_max_tokens = num_tokens_view[pos]
            batch_start = batches_ends_view[batches_count]
            batches_count += 1
            new_batch_max_tokens = tail_max_tokens

        if overflow or size_matches_with_bsz_mult:
            batches_ends_view[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0
    if batches_ends_view[batches_count] != indices_len:
        batches_count += 1
    # Memory and time-efficient split
    return np.split(indices, batches_ends[:batches_count])

# 先忽略这里


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(
                    a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key])
                )
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def batch_by_size_fn(
    indices,
    num_tokens_fn,
    max_tokens,
    max_sentences,
    bsz_mult,
):
    indices_len = indices.shape[0]
    num_tokens_vec = np.zeros(indices_len, dtype=np.int64)
    indices_view = indices
    num_tokens_vec_view = num_tokens_vec
    for pos in range(indices_len):
        num_tokens_vec[pos] = num_tokens_fn(indices_view[pos])
    return batch_by_size_vec(indices, num_tokens_vec, max_tokens,
                             max_sentences, bsz_mult,)


def _find_valid_shape(
    shapes_view,
    num_sentences,
    num_tokens,
):
    """Return index of first valid shape of -1 if none is found."""
    for i in range(shapes_view.shape[0]):
        if num_sentences <= shapes_view[i][0] and num_tokens <= shapes_view[i][1]:
            return i
    return -1


def batch_fixed_shapes_fast(
    indices,
    num_tokens_fn,
    fixed_shapes_sorted,
):
    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    indices_view = indices
    shapes_view = fixed_shapes_sorted

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        shape_idx = _find_valid_shape(shapes_view, len(batch) + 1, sample_len)
        if shape_idx == -1:
            batches.append(batch)
            batch = []
            sample_lens = []
            sample_len = 0
            shapes_view = fixed_shapes_sorted
        elif shape_idx > 0:
            # small optimization for the next call to _find_valid_shape
            shapes_view = shapes_view[shape_idx:]

        batch.append(idx)

    if len(batch) > 0:
        batches.append(batch)

    return batches
