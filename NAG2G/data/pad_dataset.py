# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import data_utils

from unicore.data import BaseWrapperDataset


class RightPadDatasetCoord(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_tokens_coords(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class RightPadDatasetCross2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_cross_2d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
