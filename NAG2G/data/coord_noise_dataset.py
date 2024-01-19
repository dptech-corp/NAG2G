      
from functools import lru_cache

import numpy as np
import torch
from unicore.data import Dictionary
from unicore.data import BaseWrapperDataset
from . import data_utils


def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


class CoordNoiseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tgt_dataset: torch.utils.data.Dataset,
        coord_gen_prob: float,
        coord_noise_prob: float,
        src_noise: float = 1.0,
        tgt_noise: float = 1.0,
        seed: int = 1,
    ):
        assert 0.0 <= coord_noise_prob <= 1.0

        self.dataset = dataset
        self.tgt_dataset = tgt_dataset
        self.coord_gen_prob = coord_gen_prob
        self.coord_noise_prob = coord_noise_prob
        self.seed = seed
        self.src_noise = src_noise
        self.tgt_noise = tgt_noise
        self.epoch = None

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        src_coord = self.dataset[index]
        tgt_coord = self.tgt_dataset[index]
        num_atoms = src_coord.shape[0]
        with data_utils.numpy_seed(self.seed, epoch, index):
            if np.random.rand() < self.coord_gen_prob:
                src_coord = np.copy(src_coord)
                noise = self.src_noise
            else:
                src_coord = np.copy(tgt_coord)
                noise = self.tgt_noise
            if np.random.rand() < self.coord_noise_prob:
                src_coord = src_coord + np.random.randn(num_atoms, 3) * noise
        R, T = get_optimal_transform(src_coord, tgt_coord)
        src_coord = src_coord @ R + T
        return {"coordinates": src_coord.astype(np.float32)}

    