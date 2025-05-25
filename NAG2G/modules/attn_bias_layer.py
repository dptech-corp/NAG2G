import torch
import torch.nn as nn
from functools import lru_cache
import numpy as np

import time


@lru_cache(maxsize=2)
def laplacian_pe_batch(A, k, idx_type=0):
    assert len(A.shape) == 3
    B, n, _ = A.shape
    if n <= k:
        assert (
            "the number of eigenvectors k must be smaller than the number of nodes n, "
            + f"{k} and {n} detected."
        )
    degree = A.sum(axis=-1)
    if k <= 0:
        return None, degree
    # get laplacian matrix as I - D^-0.5 * A * D^-0.5
    N = torch.diag_embed(degree.clip(1) ** -0.5)  # D^-1/2
    L = torch.eye(n, device=N.device).expand(B, -1, -1) - N * A * N
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal, EigVec = EigVal.real, EigVec.real
    idx_ = EigVal.argsort()
    if idx_type != 0:
        idx_ = torch.flip(idx_, [1])
    if idx_type == 0 or idx_type == 1:
        idx_ = idx_[:, 1 : k + 1]  # increasing order, skip the first eigenvector
    elif idx_type == 2:
        idx_ = idx_[:, 0: k]
    # Select top k eigenvectors and eigenvalues
    topk_eigvals_list = torch.gather(EigVal, 1, idx_)
    topk_EigVec_list = torch.gather(EigVec, 2, idx_.unsqueeze(1).expand(-1, n, -1))

    # Randomly flip signs
    rand_sign = 2 * (torch.rand(B, 1, k, device=N.device) > 0.5) - 1.0
    topk_EigVec_list = rand_sign * topk_EigVec_list
    topk_eigvals_list = topk_eigvals_list.unsqueeze(1).expand(-1, n, -1)

    laplacian_pe = torch.cat([topk_EigVec_list, topk_eigvals_list], 2)
    return laplacian_pe, degree


class seq2attn:  # (nn.Module):
    def __init__(self, min_node, sumto2, dictionary=None, want_h_degree=False, idx_type=0, use_class=False, multi_gap=False):
        self.min_node = min_node
        self.sumto2 = sumto2
        self.dictionary = dictionary
        assert dictionary is not None
        self.get_token_indexs(dictionary)
        self.want_h_degree = want_h_degree
        self.idx_type = idx_type
        self.use_class = use_class
        self.multi_gap = multi_gap
        if self.multi_gap:
            self.gap_list = np.array(
                [i for i in range(1, 10)]
                + [i * 10 for i in range(1, 10)]
                + [i * 100 for i in range(1, 6)]
            )

    def get_token_indexs(self, dictionary):
        token_categories = {
            "special": ["[CLS]", "[PAD]", "[UNK]", "[SEP]"],
            "charge": ["(charge"],
            "hydrogen": ["(H"],
            "chirality": ["(CHI_"],
            "n_radical_electrons": ["(RE"],
            "stereo": ["(STEREO"],
            "bonddir": ["(BONDDIR_"],
            "sep2": ["[SEP2]"],
            "gap": ["[gap"],
            "react_class": ["[class"],
            "bond": ["["],
            "atom": [""],
        }
        for category in token_categories.keys():
            setattr(self, category, [])
        for k, v in dictionary.indices.items():
            for category, tokens in token_categories.items():
                if any(token in k for token in tokens):
                    getattr(self, category).append(v)
                    break
        for category in token_categories.keys():
            category_list = getattr(self, category)
            if category_list:
                assert max(category_list) - min(category_list) + 1 == len(category_list)
                setattr(self, category, [min(category_list), max(category_list)])
            else:
                print(category, category_list)
            #     raise

    def new_status_get(self, seq, i_j, attn_adj_matrix, h_degree, bond_matrix):
        k = seq[:, -1].detach().cpu().numpy()
        flag_atom = np.logical_and(self.atom[0] <= k, k <= self.atom[1])
        i_j[flag_atom, 1] = i_j[flag_atom, 0]
        i_j[flag_atom, 0] = i_j[flag_atom, 0] + 1
        bond_matrix[flag_atom, :] = 0
        bond_matrix[flag_atom, i_j[flag_atom, 0]] = 1

        flag_gap = np.logical_and(self.gap[0] <= k, k <= self.gap[1])
        tmp_gap = k[flag_gap] - self.gap[0]
        if self.multi_gap:
            tmp_gap = self.gap_list[tmp_gap]
        else:
            tmp_gap += 1
        i_j[flag_gap, 1] -= tmp_gap

        flag_bond = np.logical_and(self.bond[0] <= k, k <= self.bond[1])
        flag_bond2 = np.logical_and(i_j[:, 0] >= 0, i_j[:, 1] >= 0)
        flag_bond = np.logical_and(flag_bond, flag_bond2)

        attn_adj_matrix[flag_bond, i_j[flag_bond, 0], i_j[flag_bond, 1]] = 1
        attn_adj_matrix[flag_bond, i_j[flag_bond, 1], i_j[flag_bond, 0]] = 1
        bond_matrix[flag_bond, i_j[flag_bond, 1]] = k[flag_bond] - self.bond[0] + 2
        i_j[flag_bond, 1] -= 1

        flag_h = np.logical_and(self.hydrogen[0] <= k, k <= self.hydrogen[1])
        flag_h2 = np.logical_and(
            i_j[:, 0] >= 0, h_degree[np.arange(k.shape[0]), i_j[:, 0]] == 0
        )
        flag_h = np.logical_and(flag_h, flag_h2)
        h_degree[flag_h, i_j[flag_h, 0]] = k[flag_h] - self.hydrogen[0]

    def update_status(self, seq, idx, N_node):
        device = seq.device
        seq_shape_0 = seq.shape[0]
        if seq.shape[1] == 1 or (seq.shape[1] == 2 and self.use_class is True):
            i_j = np.full((seq_shape_0, 2), -1, dtype=np.int32)
            attn_adj_matrix = np.zeros([seq_shape_0, N_node, N_node])
            h_degree = np.zeros([seq_shape_0, N_node])
            bond_matrix = np.zeros([seq_shape_0, N_node])
        else:
            i_j = self.i_j[idx]
            attn_adj_matrix = self.attn_adj_matrix[idx]
            h_degree = self.h_degree[idx]
            bond_matrix = self.bond_matrix[idx]
            if N_node > attn_adj_matrix.shape[1]:
                attn_adj_matrix = np.pad(
                    attn_adj_matrix,
                    (
                        (0, 0),
                        (0, N_node - attn_adj_matrix.shape[1]),
                        (0, N_node - attn_adj_matrix.shape[2]),
                    ),
                    mode="constant",
                )
                h_degree = np.pad(
                    h_degree, ((0, 0), (0, N_node - h_degree.shape[1])), mode="constant"
                )
                bond_matrix = np.pad(
                    bond_matrix, ((0, 0), (0, N_node - bond_matrix.shape[1])), mode="constant"
                )
        self.new_status_get(seq, i_j, attn_adj_matrix, h_degree, bond_matrix)

        self.i_j = i_j
        self.attn_adj_matrix = attn_adj_matrix
        self.h_degree = h_degree
        self.bond_matrix =  bond_matrix
        return (torch.tensor(attn_adj_matrix, device=device, dtype=torch.float),
        torch.tensor(h_degree, device=device, dtype=torch.float),
        torch.tensor(bond_matrix, device=device, dtype=torch.float)
        )

    def set_attn_bias(self, seq, idx_list):
        seq_shape = seq.shape
        device = seq.device
        degree_attn_mask = torch.zeros(
            [seq_shape[0], seq_shape[1], seq_shape[1]], device=device
        )
        bond_attn_mask = torch.zeros(
            [seq_shape[0], seq_shape[1], seq_shape[1]], device=device
        )
        laplacian_attn_mask = None
        if self.min_node > 0:
            if self.sumto2:
                laplacian_attn_mask = torch.zeros(
                    [seq_shape[0], seq_shape[1], seq_shape[1], 2], device=device
                )
            else:
                laplacian_attn_mask = torch.zeros(
                    [seq_shape[0], seq_shape[1], seq_shape[1], 2 * self.min_node],
                    device=device,
                )

        if idx_list is not None:
            degree_attn_mask[:, :-1, :-1] = self.current_degree_attn_mask[idx_list]
            bond_attn_mask[:, :-1, :-1] = self.current_bond_attn_mask[idx_list]
            if laplacian_attn_mask is not None:
                laplacian_attn_mask[:, :-1, :-1] = self.current_laplacian_attn_mask[
                    idx_list
                ]

        return degree_attn_mask, laplacian_attn_mask, bond_attn_mask

    def forward(self, seq):
        # start = time.time()
        seq_tmp = seq.cpu().detach().numpy().tolist()
        idx_list = None
        if seq.shape[1] > 2 or (seq.shape[1] > 1 and self.use_class is False):
            idx_list = [
                self.current_seq.index(hash(str(seq_tmp[i][:-1])))
                for i in range(seq.shape[0])
            ]
        self.current_seq = [hash(str(seq_tmp[i])) for i in range(seq.shape[0])]

        node_in_list = (seq >= self.atom[0]) & (seq <= self.atom[1])
        len_atoms = node_in_list.sum(-1)
        N_node = max(len_atoms.max().item(), self.min_node + 2)
        # N_node = max(len_atoms.max().item(), 250)

        batch_attn_adj_matrix, h_degree, bond_matrix = self.update_status(seq, idx_list, N_node)
        laplacian_attn_list, degree_attn_list = laplacian_pe_batch(
            batch_attn_adj_matrix, self.min_node, idx_type=self.idx_type
        )
        degree_attn_list = degree_attn_list + 1

        if self.want_h_degree:
            degree_attn_list = degree_attn_list + h_degree
        if self.min_node > 0 and self.sumto2:
            laplacian_attn_list = laplacian_attn_list.reshape(
                laplacian_attn_list.shape[0],
                laplacian_attn_list.shape[1],
                2,
                self.min_node,
            ).sum(dim=-1)

        degree_attn_mask, laplacian_attn_mask, bond_attn_mask = self.set_attn_bias(seq, idx_list)
        degree_tmp = torch.cat(
            [degree_attn_list[i, : len_atoms[i]] for i in range(seq.shape[0])], 0
        )
        degree_attn_mask[:, -1][node_in_list] = degree_tmp
        flag_pad = seq[:, -1] == self.dictionary.pad()
        degree_attn_mask[flag_pad, -1] = 0

        bond_matrix_tmp = torch.cat(
            [bond_matrix[i, : len_atoms[i]] for i in range(seq.shape[0])], 0
        )
        bond_attn_mask[:, -1][node_in_list] = bond_matrix_tmp
        bond_attn_mask[flag_pad, -1] = 0

        self.current_degree_attn_mask = degree_attn_mask
        self.current_bond_attn_mask = bond_attn_mask

        if self.min_node > 0:
            laplacian_tmp = torch.cat(
                [laplacian_attn_list[i, : len_atoms[i]] for i in range(seq.shape[0])], 0
            )
            laplacian_attn_mask[:, -1][node_in_list] = laplacian_tmp
            laplacian_attn_mask[flag_pad, -1] = 0
            self.current_laplacian_attn_mask = laplacian_attn_mask
        return {
            "decoder_degree_attn_mask": degree_attn_mask.long(),
            "decoder_laplacian_attn_mask": laplacian_attn_mask,
            "decoder_bond_attn_mask": bond_attn_mask.long(),
        }

    def forward_train(self, seq):
        for i in range(1, seq.shape[1]):
            _ = self.forward(seq[:, :i])
        result = self.forward(seq)
        return result
