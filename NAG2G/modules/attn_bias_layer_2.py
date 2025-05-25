import torch
import torch.nn as nn
import time
import numpy as np
from functools import lru_cache
from NAG2G.utils.graph_process import laplacian_pe_2
torch.set_printoptions(threshold=np.inf)

def laplacian_pe_batch_np(A, k, len_atoms, matrix_update, idx_type=0):
    len_atoms = len_atoms.detach().cpu().numpy()
    batch, N_node = A.shape[0], A.shape[1]
    lap_matrix = np.zeros([batch, N_node, 2 * k])
    degree_matrix = np.zeros([batch, N_node])
    for i in range(batch):
        if (matrix_update[i] == True) and (len_atoms[i] > 0):
            lap, degree = laplacian_pe_2(A[i], k, len_atoms[i], idx_type)
            lap_matrix[i] = lap
            degree_matrix[i] = degree
    return torch.tensor(
            lap_matrix, dtype=torch.float
        ), torch.tensor(degree_matrix, dtype=torch.float)
    # return torch.tensor(lap_matrix), torch.tensor(degree_matrix)


@lru_cache(maxsize=8)
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
    # N = torch.diag_embed(degree.clip(1) ** -0.5)  # D^-1/2
    tmp = torch.from_numpy(degree.numpy().clip(1) ** -0.5)
    N = torch.diag_embed(tmp)  # D^-1/2
    L = torch.eye(n, device=N.device).expand(B, -1, -1) - N @ A @ N
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal, EigVec = EigVal.real, EigVec.real
    idx_ = EigVal.argsort()
    if idx_type != 0:
        idx_ = torch.flip(idx_, [1])
    if idx_type == 0 or idx_type == 1:
        idx_ = idx_[:, 1 : k + 1]  # increasing order, skip the first eigenvector
    elif idx_type == 2:
        idx_ = idx_[:, 0:k]
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
    def __init__(
        self, min_node, sumto2, dictionary=None, want_h_degree=False, idx_type=0
    ):
        self.min_node = min_node
        self.sumto2 = sumto2
        self.dictionary = dictionary
        assert dictionary is not None
        self.get_token_indexs(dictionary)
        self.want_h_degree = want_h_degree
        self.idx_type = idx_type

    def get_token_indexs(self, dictionary):
        token_categories = {
            "special": ["[CLS]", "[PAD]", "[UNK]", "[SEP]"],
            "charge": ["(charge"],
            "hydrogen": ["(H"],
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
                raise

    def new_status_get(self, seq, i_j, attn_adj_matrix, h_degree):
        k = seq[:, -1].detach().cpu().numpy()
        flag_atom = np.logical_and(self.atom[0] <= k, k <= self.atom[1])
        i_j[flag_atom, 1] = i_j[flag_atom, 0]
        i_j[flag_atom, 0] = i_j[flag_atom, 0] + 1

        flag_gap = np.logical_and(self.gap[0] <= k, k <= self.gap[1])
        i_j[flag_gap, 1] -= k[flag_gap] - self.gap[0] + 1

        flag_bond = np.logical_and(self.bond[0] <= k, k <= self.bond[1])
        flag_bond2 = np.logical_and(i_j[:, 0] >= 0, i_j[:, 1] >= 0)
        flag_bond = np.logical_and(flag_bond, flag_bond2)

        attn_adj_matrix[flag_bond, i_j[flag_bond, 0], i_j[flag_bond, 1]] = 1
        attn_adj_matrix[flag_bond, i_j[flag_bond, 1], i_j[flag_bond, 0]] = 1
        i_j[flag_bond, 1] -= 1

        flag_h = np.logical_and(self.hydrogen[0] <= k, k <= self.hydrogen[1])
        flag_h2 = np.logical_and(
            i_j[:, 0] >= 0, h_degree[np.arange(k.shape[0]), i_j[:, 0]] == 0
        )
        flag_h = np.logical_and(flag_h, flag_h2)
        h_degree[flag_h, i_j[flag_h, 0]] = k[flag_h] - self.hydrogen[0]
        flag_pad = k == 0
        return flag_bond, flag_pad

    def get_previous_status(self, seq, N_node, idx):
        seq_shape_0 = seq.shape[0]
        if seq.shape[1] == 1:
            i_j = np.full((seq_shape_0, 2), -1, dtype=np.int32)
            attn_adj_matrix = np.zeros([seq_shape_0, 1, 1])
            h_degree = np.zeros([seq_shape_0, 1])
        else:
            i_j = self.i_j[idx]
            attn_adj_matrix = self.attn_adj_matrix[idx]
            h_degree = self.h_degree[idx]
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
        return i_j, attn_adj_matrix, h_degree

    def update_status(self, seq, idx, N_node):
        # device = seq.device
        i_j, attn_adj_matrix, h_degree = self.get_previous_status(seq, N_node, idx)
        matrix_update, matrix_update_pad = self.new_status_get(seq, i_j, attn_adj_matrix, h_degree)

        self.i_j = i_j
        self.attn_adj_matrix = attn_adj_matrix
        self.h_degree = h_degree
        return attn_adj_matrix, h_degree, matrix_update, matrix_update_pad
        # return (
        #     torch.tensor(attn_adj_matrix, device=device, dtype=torch.float),
        #     torch.tensor(h_degree, device=device, dtype=torch.float),
        #     matrix_update,
        # )

    def get_previous_bias(self, seq, idx_list):
        seq_shape = seq.shape
        device = seq.device
        degree_attn_mask = torch.zeros(
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
            if laplacian_attn_mask is not None:
                laplacian_attn_mask[:, :-1, :-1] = self.current_laplacian_attn_mask[
                    idx_list
                ]

        return degree_attn_mask, laplacian_attn_mask

    def forward(self, seq):
        # start = time.time()
        seq_tmp = seq.cpu().detach().numpy().tolist()
        idx_list = None
        if seq.shape[1] > 1:
            idx_list = [
                self.current_seq.index(hash(str(seq_tmp[i][:-1])))
                for i in range(seq.shape[0])
            ]
        self.current_seq = [hash(str(seq_tmp[i])) for i in range(seq.shape[0])]

        node_in_list = (seq >= self.atom[0]) & (seq <= self.atom[1])
        len_atoms = node_in_list.sum(-1)
        # N_node = max(len_atoms.max().item(), self.min_node + 2)
        # N_node = max(len_atoms.max().item(), 250)
        N_node = len_atoms.max().item()
        batch_attn_adj_matrix, h_degree, matrix_update, matrix_update_pad = self.update_status(
            seq, idx_list, N_node
        )
        degree_attn_mask, laplacian_attn_mask = self.get_previous_bias(seq, idx_list)
        # laplacian_attn_list, degree_attn_list = laplacian_pe_batch(
        #     batch_attn_adj_matrix, self.min_node, idx_type=self.idx_type
        # )
        if N_node > 0 and sum(matrix_update) != 0:
            laplacian_attn_list, degree_attn_list = laplacian_pe_batch_np(
                batch_attn_adj_matrix,
                self.min_node,
                len_atoms,
                matrix_update,
                idx_type=self.idx_type,
            )
            if self.min_node > 0:
                laplacian_attn_list = laplacian_attn_list.to(seq.device)
            degree_attn_list =  degree_attn_list.to(seq.device)
            degree_attn_list = degree_attn_list + 1
            if self.want_h_degree:
                h_degree = torch.tensor(h_degree, device=seq.device, dtype=torch.float)
                degree_attn_list = degree_attn_list + h_degree
            if self.min_node > 0 and self.sumto2:
                laplacian_attn_list = laplacian_attn_list.reshape(
                    laplacian_attn_list.shape[0],
                    laplacian_attn_list.shape[1],
                    2,
                    self.min_node,
                ).sum(dim=-1)
            degree_tmp = torch.cat(
                [degree_attn_list[i, : len_atoms[i]] for i in range(seq.shape[0])], 0
            )
            degree_attn_mask[:, -1][node_in_list] = degree_tmp
            flag_pad = seq[:, -1] == self.dictionary.pad()
            degree_attn_mask[flag_pad, -1] = 0

            if self.min_node > 0:
                laplacian_tmp = torch.cat(
                    [
                        laplacian_attn_list[i, : len_atoms[i]]
                        for i in range(seq.shape[0])
                    ],
                    0,
                )
                laplacian_attn_mask[:, -1][node_in_list] = laplacian_tmp
                laplacian_attn_mask[flag_pad, -1] = 0
        if self.min_node > 0 and laplacian_attn_mask.shape[1] > 1:
                laplacian_attn_mask[~matrix_update, -1] = laplacian_attn_mask[
                    ~matrix_update, -2]
                laplacian_attn_mask[matrix_update_pad, -1] = 0
        if degree_attn_mask.shape[1] > 1:
            degree_attn_mask[~matrix_update, -1] = degree_attn_mask[~matrix_update, -2]
            degree_attn_mask[matrix_update_pad, -1] = 0

        self.current_degree_attn_mask = degree_attn_mask
        if self.min_node > 0:
            self.current_laplacian_attn_mask = laplacian_attn_mask
        return {
            "decoder_degree_attn_mask": degree_attn_mask.long(),
            "decoder_laplacian_attn_mask": laplacian_attn_mask,
        }

    def forward_train(self, seq):
        for i in range(1, seq.shape[1]):
            _ = self.forward(seq[:, :i])
        result = self.forward(seq)
        return result
