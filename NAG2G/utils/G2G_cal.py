import sys
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from .draw_img import draw_mol
from .chemutils import add_chirality, check_atom_map
from .mol_graph_basic import error, get_InchiKey, judge_InchiKey, same_smi
import os
from multiprocessing import Pool
import time
import glob
import json
use_stereoisomerism = True
from .seq2smi import seq2smiles


def get_smiles(seq, atom_map=None):
    seq = [i for i in seq.split(" ") if i != ""]
    if "[SEP]" in seq:
        seq = seq[: seq.index("[SEP]")]
    smiles = seq2smiles(seq, atom_map, use_stereoisomerism)
    return smiles


def gen_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    # [atom.SetAtomMapNum(idx + 1) for idx, atom in enumerate(mol.GetAtoms())]
    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    smiles_remap = Chem.MolToSmiles(mol)
    return smiles_remap, atoms_map


def worker(args):
    i, context, iter_N, N_beam_search, task = args
    # product for retro, reactant for syn
    gt_smiles = context[iter_N * i].split("gt_product")[0]
    target1 = context[iter_N * i + 1].split("target")[0]  # strict
    target2 = (
        context[iter_N * i + 2].replace("[SEP]", "").split("target")[0]
    )  # unstrict
    target2 = get_smiles(target2)
    # RMK: target 1, strict; target 2, unstrict

    dup_key_list = set([None])
    gt_reactant_key = get_InchiKey(target1)

    pred_list = []
    unstrict_list = []
    strict_list = []
    nodup_list = []
    pred_nodup_list = []
    flag_strict = False
    flag_unstrict = False
    for j in range(N_beam_search):
        assert "predicted" in context[iter_N * i + 3 + j]
        str2 = context[iter_N * i + 3 + j].split("predicted")[0]

        gt_smiles_remap, atom_map = gen_map(gt_smiles)
        pred = get_smiles(str2, atom_map=atom_map)

        if flag_unstrict is False and same_smi(target2, pred):
            unstrict_list.append(1)
            flag_unstrict = True
        else:
            unstrict_list.append(0)
        if not use_stereoisomerism:
            try:
                chrial_pred = add_chirality(gt_smiles, pred, task=task)
            except:
                chrial_pred = pred
        else:
            chrial_pred = pred
        pred_key = get_InchiKey(chrial_pred)
        if pred_key not in dup_key_list:
            if judge_InchiKey(pred_key, gt_reactant_key):
                nodup_list.append(1)
            else:
                nodup_list.append(0)
            dup_key_list.add(pred_key)
            pred_nodup_list.append(chrial_pred)
        if flag_strict is False and same_smi(target1, chrial_pred):
            strict_list.append(1)
            flag_strict = True
        else:
            strict_list.append(0)

        pred_list.append(chrial_pred if same_smi(target1, chrial_pred) else pred)

    nodup_list = nodup_list + [0 for _ in range(N_beam_search - len(nodup_list))]

    # print(i, "*"*10)
    # print(target1, error(target1))
    # print(i, "*"*10)
    # for ii in pred_nodup_list:
    #     print(ii, error(ii))
    if False and np.array(strict_list).sum() == 0:
        draw_mol(
            [gt_smiles, target1, None, None, None] + pred_nodup_list,
            "./img/{}_{}.png".format(i, j),
            mols_per_row=5,
            img_size=(400, 400),
        )
    return {
        "product": gt_smiles,
        # "gt_product_remap": gt_product_remap,
        "target": [target1, target2],
        "unstrict_list": unstrict_list,
        "strict_list": strict_list,
        "pred_list": pred_list,
        "nodup_list": nodup_list,
    }


def get_context_by_one(smi_path):
    print(smi_path.split("/")[-1])
    if not os.path.exists(smi_path):
        return []
    with open(smi_path, "r") as f:
        context = f.readlines()
    print("single lines:", len(context))
    return context


def run(smi_path, save_path, N_beam_search=10, if_full=False, task="retrosynthesis"):
    print("smi_path:", smi_path, sep="\t")
    print("save_path:", save_path, sep="\t")
    print("N_beam_search:", N_beam_search, sep="\t")
    # print("I'm here!")
    # exit()

    if "{}" in smi_path:
        context = []
        files = glob.glob(smi_path.replace("{}", "*"))
        for i in files:
            context += get_context_by_one(i)
    else:
        context = get_context_by_one(smi_path)

    iter_N = N_beam_search + 3  # lines per rxn = beam_size + 3
    N_mol = int(len(context) / iter_N)
    if if_full and N_mol != 5007:
        return
    print(N_mol)
    start = time.time()
    with Pool() as pool:
        results = pool.map(
            worker,
            [
                (i, context, iter_N, N_beam_search, task)
                for i in tqdm(range(len(context) // iter_N))
            ],
        )

    target_list_all = []
    unstrict_list_all = []
    strict_list_all = []
    pred_list_all = []
    nodup_list_all = []

    if save_path is not None:
        with open(
            save_path.replace(save_path.split("/")[-1], "smiles_infer.txt"), "w"
        ) as f:
            # json.dump(i, f)
            f.write(json.dumps(results, indent=4))

    for i in results:
        target = i["target"]
        unstrict_list = i["unstrict_list"]
        strict_list = i["strict_list"]
        pred_list = i["pred_list"]
        nodup_list = i["nodup_list"]

        target_list_all.append(target)
        unstrict_list_all.append(unstrict_list)
        strict_list_all.append(strict_list)
        pred_list_all.append(pred_list)
        nodup_list_all.append(nodup_list)

    if save_path is not None:
        f = open(save_path, "w")
    else:
        f = None
        print("f is NONE!!!!")
    print(time.time() - start)
    unstrict_list_all = np.array(unstrict_list_all)
    strict_list_all = np.array(strict_list_all)
    unstrict_list_all = unstrict_list_all.sum(0)
    strict_list_all = strict_list_all.sum(0)
    nodup_list_all = np.array(nodup_list_all)
    nodup_list_all = nodup_list_all.sum(0)
    print("total", N_mol)
    print("unstrict", unstrict_list_all)
    print("unstrict", unstrict_list_all / N_mol)
    print("strict", strict_list_all)
    print("strict", strict_list_all / N_mol)
    print("nodup_list_all", nodup_list_all)
    print("nodup_list_all", nodup_list_all / N_mol)
    print("\n")

    if f is not None:
        f.write("total " + str(N_mol))
        f.write("\nunstrict " + str(unstrict_list_all))
        f.write("\nunstrict " + str(unstrict_list_all / N_mol))
        f.write("\nstrict " + str(strict_list_all))
        f.write("\nstrict " + str(strict_list_all / N_mol))
        f.write("\nnodup_list_all " + str(nodup_list_all))
        f.write("\nnodup_list_all " + str(nodup_list_all / N_mol))
        f.write("\n")

    # Accumulated
    unstrict_list_tmp = [unstrict_list_all[0]]
    for i in range(1, len(unstrict_list_all)):
        unstrict_list_tmp.append(unstrict_list_tmp[i - 1] + unstrict_list_all[i])
    unstrict_list_tmp = np.array(unstrict_list_tmp)

    strict_list_tmp = [strict_list_all[0]]
    for i in range(1, len(strict_list_all)):
        strict_list_tmp.append(strict_list_tmp[i - 1] + strict_list_all[i])
    strict_list_tmp = np.array(strict_list_tmp)

    nodup_list_tmp = [nodup_list_all[0]]
    for i in range(1, len(nodup_list_all)):
        nodup_list_tmp.append(nodup_list_tmp[i - 1] + nodup_list_all[i])
    nodup_list_tmp = np.array(nodup_list_tmp)

    print("unstrict", unstrict_list_tmp)
    print("unstrict", unstrict_list_tmp / N_mol)
    print("strict", strict_list_tmp)
    print("strict", strict_list_tmp / N_mol)
    print("nodup_list_all", nodup_list_tmp)
    print("nodup_list_all", nodup_list_tmp / N_mol)

    if f is not None:
        f.write("unstrict " + str(unstrict_list_tmp))
        f.write("\nunstrict " + str(unstrict_list_tmp / N_mol))
        f.write("\nstrict " + str(strict_list_tmp))
        f.write("\nstrict " + str(strict_list_tmp / N_mol))
        f.write("\nnodup_list_all " + str(nodup_list_tmp))
        f.write("\nnodup_list_all " + str(nodup_list_tmp / N_mol))
        f.close()


def write_failed_samples(save_path):
    with open(
        save_path.replace(save_path.split("/")[-1], "smiles_infer.txt"), "r"
    ) as fp:
        results = json.load(fp)
        # json.dump(i, f)
        # fp.write(json.dumps(results, indent=4))

    failed_samples = []
    atom_map_flags = {i: 0 for i in range(4)}
    for i, result in enumerate(results):
        (
            product,
            target,
            unstrict_list,
            strict_list,
            pred_list,
            nodup_list,
        ) = result.values()

        if unstrict_list == strict_list:
            continue

        un_idx = unstrict_list.index(1) if 1 in unstrict_list else None
        str_idx = strict_list.index(1) if 1 in strict_list else None
        if un_idx is None and str_idx is not None:
            print(unstrict_list, strict_list)
            continue
        if un_idx is not None and str_idx != un_idx:
            pred = pred_list[un_idx]
            atom_map_flag = check_atom_map(target[0], pred)
            atom_map_flags[atom_map_flag] += 1
            sample = {
                "idx": [len(failed_samples), i],
                "product": product,
                "target": target,
                "pred": pred,
                "un_str": [un_idx, str_idx],
                "atom_map_flag": atom_map_flag,
            }
            failed_samples.append(sample)

    with open(
        save_path.replace(save_path.split("/")[-1], "strict_failed_samples.json"), "w"
    ) as fp:
        json.dump(failed_samples, fp, indent=4)

    print(atom_map_flags)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise "ERROR"
    smi_path = sys.argv[1]
    N_beam_search = 10
    if len(sys.argv) >= 3:
        N_beam_search = int(sys.argv[2])
    if "--if_full" in sys.argv:
        if_full = True
    else:
        if_full = False
    if len(sys.argv) >= 5:
        score_name = sys.argv[4]
    else:
        score_name = "score"
    save_path = smi_path.replace(smi_path.split("/")[-1], score_name)

    run(smi_path, save_path, N_beam_search=N_beam_search, if_full=if_full)
