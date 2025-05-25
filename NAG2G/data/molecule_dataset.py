# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from unicore.data import BaseWrapperDataset
from numba import njit
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from NAG2G.utils.allowable_features_dict import allowable_features


def get_RSchiraltag(mol):
    chiral_type_list = ["", "S", "R"]
    atoms_chiraltag = [0 for _ in range(mol.GetNumAtoms())]
    for i, c in Chem.FindMolChiralCenters(mol):
        atoms_chiraltag[i] = chiral_type_list.index(c)
    return atoms_chiraltag


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
    # atoms_chiraltag = get_RSchiraltag(mol)
    for idx, atom in enumerate(mol.GetAtoms()):
        tmp_feature = atom_to_feature_vector(atom)
        # tmp_feature[1] = atoms_chiraltag[idx]
        atom_features_list.append(tmp_feature)
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


class MoleculeFeatureDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi_key="smi", drop_feat_prob=0.5, seed=None):
        self.dataset = dataset
        self.smi_key = smi_key
        self.drop_feat_prob = drop_feat_prob
        self.seed = seed
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        data = self.dataset[idx]
        if self.smi_key is not None:
            smiles = data[self.smi_key]
        else:
            smiles = data
            data = {}
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = smiles

        # remove atom
        mol = AllChem.AddHs(mol, addCoords=True)
        atoms_h = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
        atoms = np.array(
            [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
        )
        atoms_map = np.array(
            [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
        )
        # change AllChem.RemoveHs to AllChem.RemoveAllHs
        mol = AllChem.RemoveAllHs(mol)
        x, edge_index, edge_attr = get_graph(mol)

        # data["atoms"] = np.array(data["atoms"])
        data["atoms"] = atoms
        data["node_attr"] = x
        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["atoms_h_token"] = atoms_h
        data["atoms_token"] = atoms
        data["atoms_map"] = atoms_map
        data["smi"] = data[self.smi_key] if self.smi_key is not None else data

        data["drop_feat"] = False
        return data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


def pad_1d(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    tensor = torch.full([batch_size, pad_len], pad_value, dtype=samples[0].dtype)
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor


def pad_1d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 2
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor


def pad_2d(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    tensor = torch.full(
        [batch_size, pad_len, pad_len], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor


def pad_2d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 3
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor


def pad_attn_bias(samples, pad_len):
    batch_size = len(samples)
    pad_len = pad_len + 1
    tensor = torch.full(
        [batch_size, pad_len, pad_len], float("-inf"), dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
        tensor[i, samples[i].shape[0] :, : samples[i].shape[1]] = 0
    return tensor
