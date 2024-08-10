# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import sys
import pickle
import logging
from tqdm import tqdm
import numpy as np
from preprocess_smi_to_3d import smi2coords_3D, smi2coords_2D
from basic import (
    csv_file_read,
    get_target_order,
    rm_h_coordinates_map,
    get_atoms,
    get_canonical_smile,
    renumber_atom_maps,
)

logger = logging.getLogger(__name__)


def make_lmdb(path_smi, outputfilename):
    assert ".csv" in path_smi
    try:
        os.remove(outputfilename)
    except:
        pass

    dataset_smi = csv_file_read(path_smi)

    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)

    ii = 0
    for i in tqdm(range(len(dataset_smi))):
        result = dataset_smi[i]
        raw_string = result[2]
        target = raw_string.split(">")[-1]
        target = get_canonical_smile(target)
        reactant = raw_string.split(">")[0]
        reactant = get_canonical_smile(reactant)
        reactant = renumber_atom_maps(reactant)
        target = renumber_atom_maps(target)
        raw_string = f"{reactant}>>{target}"
        result = {}
        result["rxn_smiles"] = raw_string
        result["target_map"] = get_target_order(target, check=False, add_h=True)
        result["target_atoms"] = get_atoms(target, add_h=True)
        tmp_result_3d = smi2coords_3D(target)
        not_exist_3d = tmp_result_3d is None or len(tmp_result_3d["coordinates"]) == 0
        if not not_exist_3d:
            assert result["target_atoms"] == tmp_result_3d["atoms"]
            result["target_coordinates"] = tmp_result_3d["coordinates"].copy()
        tmp_result_2d = smi2coords_2D(target)
        not_exist_2d = tmp_result_2d is None or len(tmp_result_2d["coordinates"]) == 0
        if not_exist_2d and not_exist_3d:
            print("No 2D or 3D coordinates")
            continue
        elif not not_exist_2d:
            if not not_exist_3d:
                assert tmp_result_2d["atoms"] == result["target_atoms"]
                result["target_coordinates"] = tmp_result_2d["coordinates"].copy()
            else:
                assert tmp_result_2d["atoms"] == result["target_atoms"]
                assert (
                    result["target_coordinates"][0].shape
                    == tmp_result_2d["coordinates"][0].shape
                )
                result["target_coordinates"] += tmp_result_2d["coordinates"].copy()

        result["target_coordinates"] = np.array(result["target_coordinates"])
        if result["target_coordinates"].shape[0] > 0:
            target_atoms_tmp, target_coordinates_tmp, target_map_tmp = (
                rm_h_coordinates_map(
                    result["target_atoms"],
                    result["target_coordinates"],
                    result["target_map"],
                )
            )
            result["target_atoms"] = target_atoms_tmp
            result["target_coordinates"] = target_coordinates_tmp
            result["target_map"] = target_map_tmp
        if len(result["target_coordinates"]) > 0:
            inner_output = pickle.dumps(result, protocol=-1)
            txn_write.put(f"{ii}".encode("ascii"), inner_output)
            ii += 1
    txn_write.commit()
    env_new.close()
    print("count", ii)


if __name__ == "__main__":
    path_smi = sys.argv[1]
    outputfilename = sys.argv[2]
    make_lmdb(
        path_smi=path_smi,
        outputfilename=outputfilename,
    )
