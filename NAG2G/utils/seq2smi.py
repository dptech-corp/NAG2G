from rdkit import Chem
import random
from .features_dict import (
    chirality_dict,
    bond_type_dict,
    bond_stereo_dict,
    bond_dir_dict,
)
from .smiles_mol_trans import reconstruct_smiles


def seq2info(seq_list, use_stereoisomerism=False):
    atom_info, bond_info = [], []
    i = -1
    j = -1
    bond_flag = None
    for k in seq_list:
        if (
            (k in ["[CLS]", "[PAD]", "[UNK]", "[SEP2]"])
            or ("class" in k)
            or (not use_stereoisomerism and "(CHI" in k)
            or (not use_stereoisomerism and "(STEREO" in k)
            or (not use_stereoisomerism and "(BONDDIR_" in k)
        ):
            bond_flag = None
        elif k in ["[SEP]"]:
            break
        elif "[" not in k and "(" not in k:
            bond_flag = None
            current_atom_info = {
                "symbol": k,
                "charge": 0,
                "hydrogens": 0,
                "hydrogens_i": None,
                "chirality": Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                "n_radical_electrons": 0,
                "atommap": i + 1,
            }
            atom_info.append(current_atom_info)
            i += 1
            j = i - 1
        elif "(charge" in k:
            if bond_flag is not None:
                continue
            key = int(k[7:-1])
            if i >= 0 and current_atom_info["charge"] == 0:
                current_atom_info["charge"] = key
        elif "(H" in k:
            if bond_flag is not None:
                continue
            key = int(k[2:-1])
            if i >= 0 and current_atom_info["hydrogens"] == 0:
                current_atom_info["hydrogens"] = key
        elif "(RE" in k:
            if bond_flag is not None:
                continue
            key = int(k[3:-1])
            if i >= 0 and current_atom_info["n_radical_electrons"] == 0:
                current_atom_info["n_radical_electrons"] = key
        elif "(CHI_" in k:
            if bond_flag is not None:
                continue
            if (
                use_stereoisomerism
                and i >= 0
                and current_atom_info["chirality"]
                == Chem.rdchem.ChiralType.CHI_UNSPECIFIED
            ):
                current_atom_info["chirality"] = chirality_dict[k[1:-1]]
        elif "(STEREO" in k:
            if (
                use_stereoisomerism
                and bond_flag is not None
                and current_bond_info["stereo"] == Chem.rdchem.BondStereo.STEREONONE
            ):
                current_bond_info["stereo"] = bond_stereo_dict[k[1:-1]]
        elif "(BONDDIR_" in k:
            if (
                use_stereoisomerism
                and bond_flag is not None
                and current_bond_info["bond_dir"] == Chem.rdchem.BondDir.NONE
            ):
                key = k[9:-1]
                current_bond_info["bond_dir"] = bond_dir_dict[key]
                # if "R_" in key:
                #     current_bond_info["begin_atom"], current_bond_info["end_atom"] = bond_flag
        elif "gap" in k:
            bond_flag = None
            j -= int(k[4:-1])
        elif i >= 0 and j >= 0:
            key = k[1:-1]
            current_bond_info = {
                "begin_atom": i,
                "end_atom": j,
                "bond_type": bond_type_dict[key],
                "stereo": Chem.rdchem.BondStereo.STEREONONE,
                "bond_dir": Chem.rdchem.BondDir.NONE,
            }
  
            bond_info.append(current_bond_info)
            bond_flag = j, i
            j -= 1
    return atom_info, bond_info


def seq2smiles(seq_list, atom_map=None, use_stereoisomerism=False):
    atom_info, bond_info = seq2info(seq_list, use_stereoisomerism=use_stereoisomerism)
    try:
        if atom_map is not None:
            if isinstance(atom_map, tuple):
                atom_map = atom_map[1]
            if len(atom_map) > len(atom_info):
                atom_map = atom_map[: len(atom_info)]
            elif len(atom_map) < len(atom_info):
                atom_map = atom_map + [0 for _ in range(len(atom_info) - len(atom_map))]
        if atom_map is not None:
            for idx, i in enumerate(atom_info):
                i["atommap"] = atom_map[idx]
        reconstructed_smiles = reconstruct_smiles(atom_info, bond_info)
    except Exception as e:
        print(e)
        reconstructed_smiles = ""
    return reconstructed_smiles


def get_seq_from_mol(mol, multi_gap=True):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    seq = ["[CLS]"]
    for i, atom in enumerate(mol.GetAtoms()):
        gap_count = 0
        seq.append(atom.GetSymbol())
        key = str(atom.GetChiralTag())
        if key != "CHI_UNSPECIFIED":
            seq.append("({})".format(key))
        key = atom.GetNumRadicalElectrons()
        if key != 0:
            seq.append("(RE{})".format(key))
        key = atom.GetFormalCharge()
        if key != 0:
            seq.append("(charge{})".format(key))
        key = atom.GetNumExplicitHs()
        if key != 0:
            seq.append("(H{})".format(key))
        for j in range(i - 1, -1, -1):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                gap_count += 1
            else:
                if gap_count != 0:
                    if multi_gap:
                        for gap_tmp in gap_deal(gap_count):
                            seq.append("[gap{}]".format(gap_tmp))
                    else:
                        seq.append("[gap{}]".format(gap_count))
                    gap_count = 0

                key = bond.GetBondType()
                seq.append("[{}]".format(key))
                key = str(bond.GetStereo())
                if key != "STEREONONE":
                    seq.append("({})".format(key))
                key = str(bond.GetBondDir())
                if key not in ["NONE", "UNKNOWN"]:
                    seq.append("(BONDDIR_{})".format(key))
    seq.append("[SEP]")
    return seq


def test(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if True:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        [a.SetIsotope(0) for a in mol.GetAtoms()]
        list_ = [i for i, _ in enumerate(mol.GetAtoms())]
        random.shuffle(list_)
        mol = Chem.RenumberAtoms(mol, list_)
        smiles_refined = Chem.MolToSmiles(mol, doRandom=True)
    else:
        smiles_refined = smiles

    seq = get_seq_from_mol(mol)
    reconstructed_smiles = seq2smiles(seq, use_stereoisomerism=True)
    same = same_smi(smiles_refined, reconstructed_smiles)
    # draw_mol([smiles, smiles_refined, reconstructed_smiles], "2.png")
    # raise
    if not same:
        print("*" * 10)
        print(smiles)
        print(smiles_refined)
        print(reconstructed_smiles)
        # print(chis1, chis2)
        # draw_mol([smiles, smiles_refined, reconstructed_smiles], "2.png")
        # raise
    return same


if __name__ == "__main__":
    smiles = [
        "O=C(OCc1ccccc1)[NH:10][CH2:9][CH2:8][CH2:7][CH2:6][C@@H:5]([C:3]([O:2][CH3:1])=[O:4])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]"
    ]

    import pandas as pd
    from tqdm import tqdm

    # p = "/data/users/yaolin/NAG2G_/pistachio/pistachio_clean_dataset_filter_all/test.csv"
    p = "/data/fs_projects/NAG2G/dataset/NAG2G_/USPTO50K_raw_20230220/test.csv"
    # p = "/data/fs_projects/NAG2G/dataset/synthesis/uspto_full_bek2/test.csv"
    df = pd.read_csv(p)
    smiles = list(df["rxn_smiles"])
    smiles = [i.split(">")[0] for i in smiles]
    flag = 0
    for smi in tqdm(smiles):
        if not test(smi):
            flag += 1
    print(flag)
