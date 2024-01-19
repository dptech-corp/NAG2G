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


input_data = sys.argv[1]
output_data = sys.argv[2]
subset = sys.argv[3]

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
for i, key in enumerate(keys):
    val = input_env.begin().get(key)
    cur_id = int(np.frombuffer(key, dtype=np.int64))
    new_key = cur_id.to_bytes(4, "big")
    txn_write.put(new_key, val)
    if (i + 1) % 10000 == 0:
        txn_write.commit()
        txn_write = output_env.begin(write=True)
        print("Done {} datapoints".format(i + 1))
txn_write.commit()
output_env.close()
print("Done! ")
