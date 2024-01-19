from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import random
import torch
import time
from tqdm import tqdm
from functools import lru_cache, wraps

# from scipy.sparse.csgraph import laplacian
import scipy.sparse as sparse


# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
    ],
    "possible_degree_list": list(range(99)) + ["misc"],
    "possible_formal_charge_list": [i - 8 for i in range(17)] + ["misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "S",
        "SP2D",
        "OTHER",
        "UNSPECIFIED",
        "misc",
    ],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
        "UNSPECIFIED",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
        "misc",
    ],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def fill_adj_matrix(result):
    num_atoms = result["atoms"].shape[0]
    adj_matrix = np.full((num_atoms, num_atoms), -1, dtype=result["edge_index"].dtype)
    adj_matrix[result["edge_index"][0, :], result["edge_index"][1, :]] = result[
        "edge_attr"
    ][:, 0]
    assert np.array_equal(adj_matrix, adj_matrix.T)
    return adj_matrix


def laplacian_pe_2(A, k, idx_type):
    n = A.shape[0]
    assert n> k and k <= 0
    degree = A.sum(axis=-1)
    return None, degree


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
    charge_h_last=False
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
    if want_attn:
        N_node = max(len_atoms, min_node + 2)
        # N_node = max(len_atoms, 250)
        attn_adj_matrix = np.zeros([N_node, N_node], dtype=np.float32)
        degree_attn_list = []
        laplacian_attn_list = []
        flag_atom = np.array([0] * N_node, dtype=int)
        flag_atom_list = []
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
        list_add(token=result["atoms"][i], **dict_)
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
            if adj_matrix[i, j] == -1:
                gap_count += 1
            else:
                if gap_count != 0:
                    list_add(token="[gap{}]".format(gap_count), **dict_)
                    gap_count = 0

                key = allowable_features["possible_bond_type_list"][adj_matrix[i, j]]
                assert key != "misc"
                key = "[{}]".format(key)
                if want_attn:
                    attn_adj_matrix[i, j] = 1
                    attn_adj_matrix[j, i] = 1
                    update_dict["is_A_updated"] = True
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
        }
    else:
        return {
            "seq": seq_list,
        }


def seq2graph(
    seq_list,
    min_node=None,
    result=None,
    want_attn=False,
):
    if want_attn:
        raise
    atoms = [i for i in seq_list if "[" not in i and "(" not in i]
    atoms_charge = np.array([0 for _ in atoms])
    atoms_h = np.array([0 for _ in atoms])

    atoms = np.array(atoms)
    len_atoms = atoms.shape[0]
    adj_matrix = (np.ones([len_atoms, len_atoms]) * (-1)).astype(np.int32)
    i = -1
    j = -1
    for k in seq_list:
        if (k in ["[CLS]", "[PAD]", "[UNK]", "[SEP2]"]) or ("class" in k):
            pass
        elif k in ["[SEP]"]:
            break
        elif "[" not in k and "(" not in k:
            i += 1
            j = i - 1
        elif "(charge" in k:
            key = int(k[7:-1])
            if i >= 0 and atoms_charge[i] == 0:
                atoms_charge[i] = key
        elif "(H" in k:
            key = int(k[2:-1])
            if i >= 0 and atoms_h[i] == 0:
                atoms_h[i] = key

        elif "gap" in k:
            j -= int(k[4:-1])
        else:
            bond_type_value = allowable_features["possible_bond_type_list"].index(
                k[1:-1]
            )
            if i >= 0 and j >= 0:
                adj_matrix[i, j] = bond_type_value
                adj_matrix[j, i] = bond_type_value
            j -= 1
    if result is not None:
        assert (result["atoms"] == atoms).all()
        assert (adj_matrix == fill_adj_matrix(result)).all()
        atoms_charge_tmp = np.array(
            [
                allowable_features["possible_formal_charge_list"].index(i)
                for i in atoms_charge
            ]
        )
        atoms_h_tmp = np.array(
            [allowable_features["possible_numH_list"].index(i) for i in atoms_h]
        )
        assert (atoms_charge_tmp == result["node_attr"][:, 3]).all()
        assert (atoms_h_tmp == result["node_attr"][:, 4]).all()

    return_dict = {
        "atoms": atoms,
        "adj_matrix": adj_matrix,
        "atoms_charge": atoms_charge,
        "atoms_h": atoms_h,
    }

    return return_dict


def process_one(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = np.array([x.GetSymbol() for x in mol.GetAtoms()])
    atoms_map = np.array([x.GetAtomMapNum() for x in mol.GetAtoms()])
    node_attr, edge_index, edge_attr = get_graph(mol)
    return {
        "atoms": atoms,
        "atoms_map": atoms_map,
        "smi": smiles,
        "node_attr": node_attr,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        print(l, e)
        # raise
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr


def shuffle_graph_process(result, list_=None):
    if list_ is None:
        list_ = [i for i in range(result["atoms"].shape[0])]
        random.shuffle(list_)
    result["atoms"] = result["atoms"][list_]
    result["atoms_map"] = result["atoms_map"][list_]
    result["node_attr"] = result["node_attr"][list_]

    list_reverse = {i: idx for idx, i in enumerate(list_)}
    for i in range(result["edge_index"].shape[0]):
        for j in range(result["edge_index"].shape[1]):
            result["edge_index"][i, j] = list_reverse[result["edge_index"][i, j]]
    return result
