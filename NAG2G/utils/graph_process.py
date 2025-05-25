import torch
import numpy as np
from tqdm import tqdm
from .allowable_features_dict import allowable_features

print("*" * 50)
print("N_node = max(len_atoms, min_node + 2)")
# print("L = np.eye(n, dtype=np.float32) - N @ A @ N")
# allowable multiple choice node and edge features


def fill_adj_matrix(result):
    num_atoms = result["atoms"].shape[0]
    adj_matrix = np.full(
        (num_atoms, num_atoms, result["edge_attr"].shape[-1]),
        -1,
        dtype=result["edge_index"].dtype,
    )
    adj_matrix[result["edge_index"][0, :], result["edge_index"][1, :]] = result[
        "edge_attr"
    ]
    assert np.array_equal(adj_matrix[:, :, 0], adj_matrix[:, :, 0].T)
    return adj_matrix


def laplacian_pe_2_bek(A_tmp, k, n, idx_type, flag_rand_sign=True):
    # if n <= k:
    #     raise ValueError(
    #         f"the number of eigenvectors k must be smaller than the number of nodes n, {k} and {n} detected."
    #     )
    len_atoms = A_tmp.shape[0]
    degree = A_tmp.sum(axis=-1)
    if k <= 0:
        return None, degree
    if n is None:
        A = A_tmp
        n = A.shape[0]
        degree_ = degree
    elif n <= 0:
        return np.zeros([len_atoms, 2 * k]), np.zeros([len_atoms])
    else:
        A = A_tmp[:n, :n]
        degree_ = degree[:n]
    max_freqs = min(n - 1, k)
    # get laplacian matrix as I - D^-0.5 * A * D^-0.5
    N = np.diag(degree_.clip(1) ** -0.5)
    # N = sparse.diags(degree_.clip(1) ** -0.5, dtype=float)  # D^-1/2
    L = np.eye(n, dtype=np.float32) - N @ A @ N
    # L = laplacian(A)
    EigVal, EigVec = np.linalg.eig(L)
    EigVal, EigVec = np.real(EigVal), np.real(EigVec)

    if idx_type == 0:
        idx = EigVal.argsort()[1 : max_freqs + 1]
    elif idx_type == 1:
        idx = EigVal.argsort()[::-1][1 : max_freqs + 1]
    elif idx_type == 2:
        idx = EigVal.argsort()[::-1][0:max_freqs]
    else:
        raise
    # topk_eigvals, topk_EigVec = np.real(EigVal[idx]), np.real(EigVec[:, idx])
    topk_eigvals, topk_EigVec = EigVal[idx], EigVec[:, idx]
    # # get random flip signs
    if flag_rand_sign:
        rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
        topk_EigVec = rand_sign * topk_EigVec
    pad = k - max_freqs
    if pad > 0:
        topk_EigVec = np.pad(
            topk_EigVec, ((0, 0), (0, pad)), "constant", constant_values=0
        )
        topk_eigvals = np.concatenate([topk_eigvals, np.zeros(pad)])
    topk_eigvals = np.expand_dims(topk_eigvals, 0)
    topk_eigvals = np.repeat(topk_eigvals, topk_EigVec.shape[0], axis=0)
    result = np.concatenate([topk_EigVec, topk_eigvals], 1)
    result = np.pad(
        result, ((0, len_atoms - n), (0, 0)), mode="constant", constant_values=0
    )
    return result, degree

def laplacian_pe_2(A, k, idx_type):
    n = A.shape[0]
    if n <= k:
        raise ValueError(
            f"the number of eigenvectors k must be smaller than the number of nodes n, {k} and {n} detected."
        )
    degree = A.sum(axis=-1)
    if k <= 0:
        return None, degree
    raise
    # get laplacian matrix as I - D^-0.5 * A * D^-0.5
    N = np.diag(degree.clip(1) ** -0.5)
    # N = sparse.diags(degree.clip(1) ** -0.5, dtype=float)  # D^-1/2
    L = np.eye(n, dtype=np.float32) - N * A * N
    # L = np.eye(n, dtype=np.float32) - N @ A @ N
    # L = laplacian(A)
    EigVal, EigVec = np.linalg.eig(L)
    if idx_type == 0:
        idx = EigVal.argsort()[1 : k + 1]
    elif idx_type == 1:
        idx = EigVal.argsort()[::-1][1 : k + 1]
    elif idx_type == 2:
        idx = EigVal.argsort()[::-1][0:k]
    else:
        raise
    topk_eigvals, topk_EigVec = np.real(EigVal[idx]), np.real(EigVec[:, idx])
    # # get random flip signs
    rand_sign = 2 * (np.random.rand(k) > 0.5) - 1.0
    topk_EigVec = rand_sign * topk_EigVec
    topk_eigvals = np.expand_dims(topk_eigvals, 0)
    topk_eigvals = np.repeat(topk_eigvals, topk_EigVec.shape[0], axis=0)
    result = np.concatenate([topk_EigVec, topk_eigvals], 1)
    return result, degree


def list_add(
    degree_attn_list,
    seq_list,
    laplacian_attn_list,
    token,
    attn_adj_matrix,
    min_node,
    update_dict,
    max_seq_len,
    h_list,
    degree_h_attn_list,
    flag_atom,
    flag_atom_list,
    idx_type,
    bond_attn_list,
    bond_list,
):
    len_seq_list = len(seq_list)
    if max_seq_len is not None and len_seq_list >= max_seq_len:
        if len_seq_list > max_seq_len:
            raise
        return
    seq_list.append(token)
    if (
        attn_adj_matrix is not None
        and degree_attn_list is not None
        and laplacian_attn_list is not None
    ):
        list_add_pe(
            degree_attn_list,
            laplacian_attn_list,
            attn_adj_matrix,
            min_node,
            update_dict,
            idx_type=idx_type,
        )
    if h_list is not None and degree_h_attn_list is not None:
        degree_h_attn_list.append(h_list.copy())
    if flag_atom is not None and flag_atom_list is not None:
        flag_atom_list.append(flag_atom.copy())
    if bond_list is not None and bond_attn_list is not None:
        bond_attn_list.append(bond_list.copy())


def list_add_pe(
    degree_attn_list,
    laplacian_attn_list,
    attn_adj_matrix,
    min_node,
    update_dict,
    idx_type,
):
    if update_dict["is_A_updated"]:
        result, degree = laplacian_pe_2(attn_adj_matrix, min_node, idx_type=idx_type)
        update_dict["last_result"], update_dict["last_degree"] = result, degree
        update_dict["is_A_updated"] = False

    else:
        result, degree = update_dict["last_result"], update_dict["last_degree"]
    # degree_attn_list.append(attn_adj_matrix.sum(axis=-1))
    degree_attn_list.append(degree)
    if min_node > 0:
        laplacian_attn_list.append(result)


def gap_deal(gap):
    assert gap < 600
    digits = [int(d) * 10**i for i, d in enumerate(str(gap)[::-1])][::-1]
    return [i for i in digits if i != 0]


def graph2seq_process(
    result,
    class_idx,
    min_node,
    want_attn=False,
    want_charge_h=True,
    max_seq_len=None,
    sumto2=True,
    use_sep2=False,
    want_h_degree=False,
    idx_type=0,
    charge_h_last=False,
    multi_gap=False,
    use_stereoisomerism=False,
    want_re=False,
):
    if charge_h_last:
        raise
    update_dict = {
        "is_A_updated": True,
        "last_result": None,
        "last_degree": None,
    }
    len_atoms = result["atoms"].shape[0]

    seq_list = []
    attn_adj_matrix = None
    degree_attn_list = None
    laplacian_attn_list = None
    h_list = None
    degree_h_attn_list = None
    flag_atom = None
    flag_atom_list = None
    bond_attn_list = None
    bond_list = None
    if want_attn:
        N_node = max(len_atoms, min_node + 2)
        # N_node = max(len_atoms, 250)
        attn_adj_matrix = np.zeros([N_node, N_node], dtype=np.float32)
        degree_attn_list = []
        laplacian_attn_list = []
        flag_atom = np.array([0] * N_node, dtype=int)
        flag_atom_list = []
        bond_attn_list = []
        bond_list = np.array([0] * N_node, dtype=int)
        if want_h_degree:
            h_list = np.array([0] * N_node, dtype=int)
            degree_h_attn_list = []

    dict_ = {
        "degree_attn_list": degree_attn_list,
        "seq_list": seq_list,
        "laplacian_attn_list": laplacian_attn_list,
        "attn_adj_matrix": attn_adj_matrix,
        "min_node": min_node,
        "update_dict": update_dict,
        "max_seq_len": max_seq_len,
        "h_list": h_list,
        "degree_h_attn_list": degree_h_attn_list,
        "flag_atom": flag_atom,
        "flag_atom_list": flag_atom_list,
        "idx_type": idx_type,
        "bond_attn_list": bond_attn_list,
        "bond_list": bond_list,
    }

    list_add(token="[CLS]", **dict_)
    if class_idx is not None:
        list_add(token="[class{}]".format(class_idx), **dict_)
    adj_matrix = fill_adj_matrix(result)
    map_flag = True
    for i in range(len_atoms):
        if use_sep2 and map_flag and result["atoms_map"][i] == 0:
            map_flag = False
            list_add(token="[SEP2]", **dict_)
        if flag_atom is not None:
            flag_atom[i] = 1
        if want_attn:
            bond_list = np.array([0] * N_node, dtype=int)
            bond_list[i] = 1
        list_add(token=result["atoms"][i], **dict_)
        if result["atoms"][i] == "NoneAtom":
            continue
        if use_stereoisomerism:
            key = allowable_features["possible_chirality_list"][
                result["node_attr"][i][1]
            ]
            if key != "CHI_UNSPECIFIED":
                key = "({})".format(key)
                list_add(token=key, **dict_)
        if want_re:
            key = allowable_features["possible_number_radical_e_list"][
                result["node_attr"][i][5]
            ]
            if key != 0:
                key = "(RE{})".format(key)
                list_add(token=key, **dict_)
        if want_charge_h:
            key = allowable_features["possible_formal_charge_list"][
                result["node_attr"][i][3]
            ]
            assert key != "misc"
            if key != 0:
                key = "(charge{})".format(key)
                list_add(token=key, **dict_)

            key = allowable_features["possible_numH_list"][result["node_attr"][i][4]]
            assert key != "misc"
            if key != 0:
                if h_list is not None and h_list[i] == 0:
                    h_list[i] = key
                key = "(H{})".format(key)
                list_add(token=key, **dict_)
        if i == 0:
            continue
        gap_count = 0
        for j in range(i - 1, -1, -1):
            if adj_matrix[i, j, 0] == -1:
                gap_count += 1
            else:
                if gap_count != 0:
                    if multi_gap:
                        for gap_tmp in gap_deal(gap_count):
                            list_add(token="[gap{}]".format(gap_tmp), **dict_)
                    else:
                        list_add(token="[gap{}]".format(gap_count), **dict_)
                    gap_count = 0

                key = allowable_features["possible_bond_type_list"][adj_matrix[i, j, 0]]
                assert key != "misc"
                key = "[{}]".format(key)
                if want_attn:
                    attn_adj_matrix[i, j] = 1
                    attn_adj_matrix[j, i] = 1
                    update_dict["is_A_updated"] = True
                if want_attn:
                    bond_list[i] = adj_matrix[i, j, 0] + 2
                list_add(token=key, **dict_)
                if use_stereoisomerism:
                    key = allowable_features["possible_bond_stereo_list"][
                        adj_matrix[i, j, 1]
                    ]
                    if key != "STEREONONE":
                        key = "({})".format(key)
                        list_add(token=key, **dict_)
                    key = allowable_features["possible_bond_dir_list"][adj_matrix[i, j, 3]]
                    if key not in ["NONE", "UNKNOWN"]:
                        key = "(BONDDIR_{})".format(key)
                        list_add(token=key, **dict_)
    list_add(token="[SEP]", **dict_)

    if want_attn:
        node_in_list = ["[" not in i and "(" not in i for i in seq_list]
        flag_atom_list = np.stack(flag_atom_list)
        degree_attn_list = np.stack(degree_attn_list) + flag_atom_list
        if want_h_degree:
            degree_h_attn_list = np.stack(degree_h_attn_list)
            degree_attn_list = degree_attn_list + degree_h_attn_list
        degree_attn_list = degree_attn_list[:, : sum(node_in_list)]
        degree_attn_mask = np.zeros([len(seq_list), len(seq_list)])
        degree_attn_mask[:, node_in_list] = degree_attn_list
        # bond_attn_mask = np.zeros([len(seq_list), len(seq_list)])
        # bond_attn_list = bond_attn_list[:, : sum(node_in_list)]
        # bond_attn_mask[:, node_in_list] = bond_attn_list
        if min_node > 0:
            laplacian_attn_list = np.stack(laplacian_attn_list)
            laplacian_attn_list[flag_atom_list == 0] = 0
            laplacian_attn_list = laplacian_attn_list[:, : sum(node_in_list)]
            laplacian_attn_mask = np.zeros([len(seq_list), len(seq_list), min_node * 2])
            laplacian_attn_mask[:, node_in_list, :] = laplacian_attn_list
            laplacian_attn_mask = torch.tensor(laplacian_attn_mask, dtype=torch.float)
            if sumto2:
                laplacian_attn_mask = laplacian_attn_mask.reshape(
                    len(seq_list), len(seq_list), 2, min_node
                ).sum(dim=-1)
        else:
            laplacian_attn_mask = None
        return {
            "seq": seq_list,
            "degree_attn_mask": torch.tensor(degree_attn_mask).long(),
            "laplacian_attn_mask": laplacian_attn_mask,
            # "bond_attn_mask": torch.tensor(bond_attn_mask).long(),
        }
    else:
        return {
            "seq": seq_list,
        }


if __name__ == "__main__":
    smiles = "[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][n:11]2[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18])[cH:19]1"
    # smiles = "CC(C)(C)OC(=O)O[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18].[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][nH:11]2)[cH:19]1"
    smiles = [
        "[F:3][c:4]1[cH:5][cH:6][c:7](-[c:10]2[c:11]([C:37](=[O:38])[NH:42][CH3:41])[c:12]3[c:13]([n:14][cH:15][c:16](-[c:18]4[cH:19][c:20]([C:24]([NH:25][C:26]([CH3:27])([CH3:28])[c:29]5[cH:30][cH:31][cH:32][cH:33][cH:34]5)=[O:35])[cH:21][cH:22][cH:23]4)[cH:17]3)[o:36]2)[cH:8][cH:9]1"
    ]

    min_node = 30
    from mol_graph_basic import graph2mol, same_smi, error, get_canonical_smile
    from smiles2feature import process_one

    count = 0
    for i in tqdm(smiles):
        # i = smiles
        # print(i)
        result = process_one(i)
        tmp = graph2seq_process(
            result,
            class_idx=None,
            min_node=min_node,
            want_attn=True,
            want_h_degree=True,
        )
        seq, degree_attn_mask, laplacian_attn_mask = (
            tmp["seq"],
            tmp["degree_attn_mask"],
            tmp["laplacian_attn_mask"],
        )
        print(seq)
        tmp = seq2graph(seq, min_node=min_node, result=result)
        adjacency_matrix = tmp["adj_matrix"] + 1
        adjacency_matrix[adjacency_matrix == 4] = 12
        atoms = tmp["atoms"]
        atoms_charge = tmp["atoms_charge"].tolist()
        atom_h_number = tmp["atoms_h"].tolist()

        b = graph2mol(
            adjacency_matrix=adjacency_matrix,
            atoms=atoms,
            atoms_charge=atoms_charge,
            atom_h_number=atom_h_number,
        )
        if not same_smi(get_canonical_smile(i, False), b):
            # if error(b) is False:
            print(i)
            print(b)
            count += 1
    print(count)
    # degree_attn_mask2, laplacian_attn_mask2 = (
    #     tmp["degree_attn_mask"],
    #     tmp["laplacian_attn_mask"],
    # )
    # print((degree_attn_mask == degree_attn_mask2).all())
    # print((laplacian_attn_mask == laplacian_attn_mask2).all())
