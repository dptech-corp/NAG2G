import logging
import os
import sys
import pickle
import torch
import lmdb
import gzip
import numpy as np
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
from multiprocessing import Pool
from tqdm import tqdm


input_data = sys.argv[1]
output_data = sys.argv[2]
subset = sys.argv[3]
top = int(sys.argv[4])


def load(num_pickles=8, top=-1):
    pred_pos_dict = {}
    for i in range(num_pickles):
        pickle_path = os.path.join(output_data, subset + "_{}.pkl".format(i))
        pickle_data = pickle.load(open(pickle_path, "rb"))
        for x in pickle_data:
            id, pos_pred, plddt = x
            for j in range(len(id)):
                cur_id = int(id[j])
                if cur_id not in pred_pos_dict:
                    pred_pos_dict[cur_id] = []
                pred_pos_dict[cur_id].append([pos_pred[j], plddt[j]])
    if top > 0:
        top_pred_pos_dict = {}
        for key in pred_pos_dict:
            cur_list = pred_pos_dict[key]
            cur_list.sort(key=lambda x: x[1], reverse=True)
            top_pred_pos_dict[key] = [x[0] for x in cur_list[:top]]
        return top_pred_pos_dict
    else:
        top_pred_pos_dict = {}
        for key in pred_pos_dict:
            cur_list = pred_pos_dict[key]
            top_pred_pos_dict[key] = [x[0] for x in cur_list]
        return top_pred_pos_dict


pred_pos_dict = load(top=top)

split_path = os.path.join(input_data, subset + ".lmdb")
input_env = lmdb.open(
    split_path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)
with input_env.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))
os.system("mkdir -p {}".format(output_data))
save_path = os.path.join(output_data, subset + ".lmdb")
os.system("rm -f {}".format(save_path))

output_env = lmdb.open(
    save_path,
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = output_env.begin(write=True)
print("start to write lmdb")


def process(key):
    datapoint_pickled = input_env.begin().get(key)
    data = pickle.loads(gzip.decompress(datapoint_pickled))
    cur_id = int.from_bytes(key, "big")  # int(np.frombuffer(key, dtype=np.int64))
    old_pos = data["input_pos"]
    num_atoms = old_pos[0].shape[0]
    new_pos = [x[:num_atoms, :] for x in pred_pos_dict[cur_id]]
    # assert len(old_pos) == len(new_pos)
    data["input_pos"] = new_pos
    val = gzip.compress(pickle.dumps(data))
    return key, val


i = 0
with Pool(64) as pool:
    for ret in tqdm(pool.imap_unordered(process, keys), total=len(keys)):
        key, val = ret
        txn_write.put(key, val)
        if (i + 1) % 10000 == 0:
            txn_write.commit()
            txn_write = output_env.begin(write=True)
        i += 1

txn_write.commit()
output_env.close()
print("Done inference! ")
