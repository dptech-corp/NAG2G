# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils
from numba import njit
from .collator import *


@njit
def floyd_warshall(M, path):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 510

    for i in range(n):
        M[i, i] = 0

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj
                    path[i, j] = k

    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                path[i, j] = 510
                M[i, j] = 510
    return M, path


def get_all_edges(path, i, j, max_dist):
    if max_dist <= 0:
        return []
    k = path[i][j]
    if k == -1:
        return []
    else:
        left = get_all_edges(path, i, k, max_dist - 1)
        if len(left) + 1 >= max_dist:
            return left + [k]
        right = get_all_edges(path, k, j, max_dist - len(left) - 1)
        return left + [k] + right


# @njit
def gen_edge_input(max_dist, path_copy, edge_feat):

    (nrows, ncols) = path_copy.shape
    assert nrows == ncols
    n = nrows
    max_dist_copy = max_dist

    edge_fea_all = -1 * np.ones(
        [n, n, max_dist_copy, edge_feat.shape[-1]], dtype=np.int32
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = (
                [i] + get_all_edges(path_copy, i, j,
                                    max_dist=max_dist_copy + 1) + [j]
            )
            num_path = min(len(path) - 1, max_dist_copy)
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat[path[k], path[k + 1], :]

    return edge_fea_all


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[-1] if len(x.shape) > 1 else 1
    feature_offset = 1 + \
        np.arange(0, feature_num * offset, offset, dtype=np.int32)
    x = x + feature_offset
    return x


def convert_to_single_emb_torch(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item, want_edge_input=True):
    edge_attr, edge_index, x = (
        item["edge_attr"],
        item["edge_index"],
        item["node_attr"],
    )
    N = x.shape[0]
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    path = np.full([N, N], -1, dtype=np.int32)
    shortest_path_result, path = floyd_warshall(adj, path)

    if want_edge_input:
        max_dist = min(np.amax(shortest_path_result), 6)
        edge_input = gen_edge_input(max_dist, path, attn_edge_type)

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    feat = {}
    feat["x"] = torch.from_numpy(x).long()
    feat["attn_bias"] = attn_bias
    feat["attn_edge_type"] = torch.from_numpy(attn_edge_type).long()
    feat["spatial_pos"] = spatial_pos
    feat["in_degree"] = torch.from_numpy(degree).long().view(-1)
    feat["out_degree"] = feat["in_degree"]  # for undirected graph
    if want_edge_input:
        feat["edge_input"] = torch.from_numpy(edge_input).long()
    else:
        feat["edge_input"] = None

    return feat


class GraphFeatures(BaseWrapperDataset):
    def __init__(self, dataset, pos_dataset, want_edge_input=True, add_len=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.pos_dataset = pos_dataset
        self.want_edge_input = want_edge_input
        self.add_len = add_len
        if self.add_len < 0:
            self.add_len = 0

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        feat = preprocess_item(data, want_edge_input=self.want_edge_input)
        if self.pos_dataset is not None:
            pos = self.pos_dataset[idx]
            feat["pos"] = torch.from_numpy(pos)
        else:
            feat["pos"] = torch.zeros(
                [feat["x"].shape[0], 3], dtype=torch.float)
        return feat

    def collater(self, items):
        multi_hop_max_dist = 5
        spatial_pos_max = 1024
        if self.want_edge_input:
            items = [
                (
                    item["attn_bias"],
                    item["attn_edge_type"],
                    item["spatial_pos"],
                    item["in_degree"],
                    item["out_degree"],
                    item["x"],
                    item["edge_input"][:, :, :multi_hop_max_dist, :],
                    item["pos"],
                )
                for item in items
            ]
        else:
            items = [
                (
                    item["attn_bias"],
                    item["attn_edge_type"],
                    item["spatial_pos"],
                    item["in_degree"],
                    item["out_degree"],
                    item["x"],
                    None,
                    item["pos"],
                )
                for item in items
            ]
        (
            attn_biases,
            attn_edge_types,
            spatial_poses,
            in_degrees,
            out_degrees,
            xs,
            edge_inputs,
            poses,
        ) = zip(*items)

        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float(
                "-inf"
            )
        max_node_num = max(i.size(0) for i in xs)
        max_node_num = max_node_num + self.add_len
        max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
        if self.want_edge_input:
            max_dist = max(i.size(-2) for i in edge_inputs)
            edge_input = torch.cat(
                [
                    pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist)
                    for i in edge_inputs
                ]
            )
        else:
            edge_input = None
        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        )
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                              for i in in_degrees])

        pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

        node_type_edges = []
        for idx in range(len(items)):

            node_atom_type = items[idx][5][:, 0]
            n_nodes = items[idx][5].shape[0]
            node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
            node_atom_i = pad_spatial_pos_unsqueeze(
                node_atom_i, max_node_num
            ).unsqueeze(-1)
            node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
            node_atom_j = pad_spatial_pos_unsqueeze(
                node_atom_j, max_node_num
            ).unsqueeze(-1)
            node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
            node_atom_edge = convert_to_single_emb_torch(node_atom_edge)

            node_type_edges.append(node_atom_edge.long())
        node_type_edge = torch.cat(node_type_edges)
        if not self.want_edge_input:
            edge_input = attn_edge_type
        return dict(
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,
            edge_input=edge_input,
            pos=pos,
            node_type_edge=node_type_edge,
        )


class ShortestPathDataset(BaseWrapperDataset):
    def __init__(self, dataset, has_bos=True, has_eos=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.has_bos = has_bos
        self.has_eos = has_eos

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_atoms = data["atoms"].shape[0]
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        adj = np.full(
            (num_atoms, num_atoms),
            510,
            dtype=np.int,
        )
        edge_index = data["edge_index"]
        adj[edge_index[0, :] + offset, edge_index[1, :] + offset] = 1
        shortest_path_result, _ = floyd_warshall(adj)
        # shortest_path_result[shortest_path_result > 510] = 510
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        return spatial_pos


class DegreeDataset(BaseWrapperDataset):
    def __init__(self, dataset, has_bos=True, has_eos=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.has_bos = has_bos
        self.has_eos = has_eos

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_atoms = data["atoms"].shape[0]
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        adj = np.full(
            (num_atoms, num_atoms),
            0,
            dtype=np.int,
        )
        edge_index = data["edge_index"]
        adj[edge_index[0, :] + offset, edge_index[1, :] + offset] = 1
        # +1 for padding
        degree = np.sum(adj, axis=1) + 1
        return torch.from_numpy(degree).long()


def collate_1d_features(
    values,
    pad_idx,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    v = values[0]
    size = max(v.size(0) for v in values)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, v.shape[-1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][: len(v), :],
        )
    return res


class AtomFeatDataset(BaseWrapperDataset):
    def __init__(
        self, dataset, num_features=8, num_vals=16, has_bos=True, has_eos=True
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_features = num_features
        self.num_vals = num_vals
        self.has_bos = has_bos
        self.has_eos = has_eos

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_atoms = data["atoms"].shape[0]
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        feat = np.full(
            (num_atoms, self.num_features),
            1,
            dtype=np.int,
        )
        node_attr = data["node_attr"]
        # skip first dimension
        feat[offset: offset + node_attr.shape[0], :] = node_attr[:, 1:] + 2
        for i in range(self.num_features):
            feat[:, i] += i * self.num_vals
        return torch.from_numpy(feat).long()

    def collater(self, samples):
        return collate_1d_features(samples, 0, pad_to_multiple=8)


def collate_2d_features(
    values,
    pad_idx,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    v = values[0]
    size = max(v.size(0) for v in values)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size, v.shape[-1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][: len(v), : len(v), :],
        )
    return res


class BondDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        num_features=4,
        num_vals=8,
        has_bos=True,
        has_eos=True,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_features = num_features
        self.num_vals = num_vals
        self.has_bos = has_bos
        self.has_eos = has_eos

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_atoms = data["atoms"].shape[0]
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        edge_feat = np.full(
            (num_atoms, num_atoms, self.num_features),
            0,
            dtype=np.int,
        )
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        # no connected
        edge_feat[:, :, 0] = 1
        # self connected
        for i in range(num_atoms):
            edge_feat[i, i, 0] = 2
        # bond connected
        edge_feat[edge_index[0, :] + offset, edge_index[1, :] + offset, 0] = 3
        # other bond features
        edge_feat[edge_index[0, :] + offset, edge_index[1, :] + offset, 1:] = (
            edge_attr + 1
        )
        for i in range(self.num_features):
            # add offset
            edge_feat[:, :, i] += self.num_vals * i
        return torch.from_numpy(edge_feat).long()

    def collater(self, samples):
        return collate_2d_features(samples, 0, pad_to_multiple=8)
