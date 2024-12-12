import random
import pandas as pd
from NAG2G.utils.graph_process import (
    process_one,
    shuffle_graph_process,
    graph2seq_process,
)

from unicore.data import UnicoreDataset, BaseWrapperDataset
from functools import lru_cache
import numpy as np


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
        return {
            "reactant_smiles": reactant_smiles,
            "product_smiles": product_smiles,
        }


class SmilesDataset_2(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        result = self.dataset[idx]
        return {
            "reactant_smiles": result["target"],
            "product_smiles": result["input"],
        }


class GraphormerDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        smiles = self.dataset[index]
        result = process_one(smiles)
        return result


class ShuffleGraphormerDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        result = self.dataset[index]
        result = shuffle_graph_process(result)
        return result


class ReorderGraphormerDataset(BaseWrapperDataset):
    def __init__(self, product_dataset, reactant_dataset, align_base="product"):
        super().__init__(product_dataset)
        self.reactant_dataset = reactant_dataset
        self.align_base = align_base
        self.set_epoch(None)
    
    def get_list(atoms_map_product, atoms_map_reactant):
        atoms_map_reactant_dict = {
            atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
        }

        tmp = [atoms_map_reactant_dict[i] for i in atoms_map_product if i in atoms_map_reactant_dict]
        
        all_indices = set(range(len(atoms_map_reactant)))
        missing_indices = list(all_indices - set(tmp))
        
        list_reactant = tmp + missing_indices
        
        return list_reactant

    def get_list(self, atoms_map_product, atoms_map_reactant):
        if self.align_base == "reactant":
            mask = atoms_map_reactant != 0
            orders = np.array([i for i in range(len(atoms_map_reactant))])
            list_reactant = np.concatenate([orders[mask], orders[~mask]], 0)
            atoms_map_product_dict = {
                atoms_map_product[i]: i for i in range(len(atoms_map_product))
            }
            list_product = [atoms_map_product_dict[i] for i in atoms_map_reactant[mask]]
        elif self.align_base == "product":
            list_product = None
            # atoms_map_reactant_dict = {
            #     atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
            # }
            # tmp = [atoms_map_reactant_dict[i] for i in atoms_map_product]
            # orders = np.array([i for i in range(len(atoms_map_reactant))])
            # mask = atoms_map_reactant != 0
            # list_reactant = np.concatenate([tmp, orders[~mask]], 0)

            atoms_map_reactant_dict = {
            atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
            }

            tmp = [atoms_map_reactant_dict[i] for i in atoms_map_product if i in atoms_map_reactant_dict]
            
            all_indices = set(range(len(atoms_map_reactant)))
            missing_indices = list(all_indices - set(tmp))
            
            list_reactant = tmp + missing_indices

        else:
            raise
        return list_product, list_reactant

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
        result = self.dataset[index]
        if self.class_dataset is not None:
            class_idx = self.class_dataset[index]
        else:
            class_idx = None
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
        )
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
