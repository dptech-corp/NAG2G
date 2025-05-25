import pandas as pd
from NAG2G.utils.graph_process import graph2seq_process
from NAG2G.utils.smiles2feature import process_one
from unicore.data import UnicoreDataset, BaseWrapperDataset
from functools import lru_cache
import numpy as np
from rdkit import Chem


debug = False
if debug:
    from NAG2G.utils.graph_process import get_seq_from_mol
    from NAG2G.utils.smiles_mol_trans import (
        extract_info_from_smiles,
        convert_atom_info,
        check,
        same_smi,
        reconstruct_smiles,
        Chem,
    )
    from NAG2G.utils.seq2smi import seq2info


def rearrange_and_pad(arr, index_list, pad_value=0):
    if len(arr) == len(index_list):
        return arr[index_list]
    result = np.array([arr[idx] if idx != -1 else pad_value for idx in index_list])
    return result


def shuffle_graph_process(result, list_):
    # if list_ is None:
    #     list_ = [i for i in range(result["atoms"].shape[0])]
    #     random.shuffle(list_)
    result_keys = result.keys()
    for i in [
        ("atoms", "NoneAtom"),
        ("atoms_map", 0),
        ("node_attr", np.array([0 for _ in range(9)])),
        ("atoms_token", 0),
    ]:
        key, pad = i
        if key in result_keys:
            result[key] = rearrange_and_pad(result[key], list_, pad_value=pad)
            # result[key] = result[key][list_]
    list_reverse = {i: idx for idx, i in enumerate(list_)}
    for i in range(result["edge_index"].shape[0]):
        for j in range(result["edge_index"].shape[1]):
            result["edge_index"][i, j] = list_reverse[result["edge_index"][i, j]]
    return result


class CsvGraphormerDataset(UnicoreDataset):
    def __init__(self, path):
        self.path = path
        self.csv_file = pd.read_csv(self.path)

    def __len__(self):
        return len(self.csv_file)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return dict(self.csv_file.iloc[idx])


class SmilesDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        tmp_smiles = self.dataset[idx]
        tmp_smiles = tmp_smiles.split(">")
        reactant_smiles = tmp_smiles[0]
        product_smiles = tmp_smiles[2]
        product_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(product_smiles))
        return {
            "reactant_smiles": reactant_smiles,
            "product_smiles": product_smiles,
        }


class GraphormerDataset(BaseWrapperDataset):
    def __init__(self, dataset, use_stereoisomerism):
        super().__init__(dataset)
        self.set_epoch(None)
        self.use_stereoisomerism = use_stereoisomerism

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        smiles = self.dataset[index]
        try:
            result = process_one(smiles)
            if self.use_stereoisomerism:
                assert result["edge_attr"].shape[-1] == 4
            else:
                result["edge_attr"] = result["edge_attr"][:, :-1]
                assert result["edge_attr"].shape[-1] == 3
        except Exception as e:
            print(e)
            print("process_one", smiles)
            raise
        return result


class ReorderGraphormerDataset(BaseWrapperDataset):
    def __init__(
        self,
        product_dataset,
        reactant_dataset,
        align_base="product",
        task_type="retrosynthesis",
    ):
        super().__init__(product_dataset)
        self.reactant_dataset = reactant_dataset
        self.align_base = align_base
        self.task_type = task_type
        self.set_epoch(None)

    def get_list(self, atoms_map_product, atoms_map_reactant):
        if self.task_type == "synthesis":
            assert 0 not in atoms_map_reactant
            # atoms_map_product = [i for i in atoms_map_product if i != 0]
        else:
            assert 0 not in atoms_map_product

        assert self.align_base == "product"
        atoms_map_reactant_dict = {
            atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
        }
        tmp = [atoms_map_reactant_dict[i] if i != 0 else -1 for i in atoms_map_product]
        orders = np.array([i for i in range(len(atoms_map_reactant))])
        mask = atoms_map_reactant != 0
        list_reactant = np.concatenate([tmp, orders[~mask]], 0)
        return None, list_reactant

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.reactant_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        product = self.dataset[index]
        reactant = self.reactant_dataset[index]
        try:
            list_product, list_reactant = self.get_list(
                product["atoms_map"], reactant["atoms_map"]
            )
        except:
            raise
            list_product, list_reactant = None, None
        if list_product is not None:
            product = shuffle_graph_process(product, list_=list_product)
        if list_reactant is not None:
            reactant = shuffle_graph_process(reactant, list_=list_reactant)
        return {"reactant": reactant, "product": product}


class SeqGraphormerDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        class_dataset,
        min_node,
        want_attn,
        want_charge_h,
        max_seq_len=None,
        sumto2=True,
        use_sep2=False,
        want_h_degree=False,
        idx_type=0,
        charge_h_last=False,
        multi_gap=False,
        use_stereoisomerism=False,
        want_re=False
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.class_dataset = class_dataset
        self.min_node = min_node
        self.want_attn = want_attn
        self.want_charge_h = want_charge_h
        self.max_seq_len = max_seq_len
        self.epoch = None
        self.sumto2 = sumto2
        self.use_sep2 = use_sep2
        self.want_h_degree = want_h_degree
        self.idx_type = idx_type
        self.charge_h_last = charge_h_last
        self.multi_gap = multi_gap
        self.use_stereoisomerism = use_stereoisomerism
        self.want_re = want_re
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        if self.class_dataset is not None:
            self.class_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        global debug
        result = self.dataset[index]
        if self.class_dataset is not None:
            class_idx = self.class_dataset[index]
        else:
            class_idx = None
        if debug:
            pass
            mol = result["smi"]
            # af1, bf1 = result["node_attr"], result["edge_attr"]
            # edge_index = result["edge_index"]
        result = graph2seq_process(
            result,
            class_idx,
            self.min_node,
            want_attn=self.want_attn,
            want_charge_h=self.want_charge_h,
            max_seq_len=self.max_seq_len,
            sumto2=self.sumto2,
            use_sep2=self.use_sep2,
            want_h_degree=self.want_h_degree,
            idx_type=self.idx_type,
            charge_h_last=self.charge_h_last,
            multi_gap=self.multi_gap,
            use_stereoisomerism=self.use_stereoisomerism,
            want_re=self.want_re,
        )

        if debug:
            seq = get_seq_from_mol(mol)
            if " ".join(result["seq"]) != " ".join(seq):
                print(seq)
                print(result["seq"])
                raise
            # tmp = seq2info(result["seq"], use_stereoisomerism=True)
            # af2, bf2 = convert_atom_info(*tmp)
            # checker = check(af1, af2, bf1, bf2, edge_index)
            # if checker is not True:
            #     print(2, checker)
            # raise
        return result


class ReorderCoordDataset(BaseWrapperDataset):
    def __init__(
        self,
        raw_coord_dataset,
        map_coord_dataset,
        product_dataset,
    ):
        super().__init__(raw_coord_dataset)
        self.product_dataset = product_dataset
        self.map_coord_dataset = map_coord_dataset
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.product_dataset.set_epoch(epoch)
        self.map_coord_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        result = self.dataset[index]
        old_map = self.map_coord_dataset[index]
        new_map = self.product_dataset[index]["atoms_map"]
        old_map_dict = {old_map[i]: i for i in range(len(old_map))}
        orders = [old_map_dict[i] for i in new_map]
        return result[orders]
