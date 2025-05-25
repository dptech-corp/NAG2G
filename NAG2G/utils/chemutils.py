# Modified from https://github.com/wengong-jin/iclr19-graph2graph
import os
import json
import datetime
import os
import json
import datetime
import rdkit.Chem as Chem
from rdchiral.chiral import copy_chirality
from rdkit.Chem import SanitizeMol, SanitizeFlags
from rdkit.Chem.AllChem import AssignStereochemistry


def canonicalize(smiles, add_atom_num=False):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(e)
        return smiles

    if tmp is None:
        print("wrong smiles: %s" % (smiles))
        return smiles

    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp("molAtomMapNumber") for a in tmp.GetAtoms()]
    smiles = Chem.MolToSmiles(tmp)

    if add_atom_num:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return smiles


def get_mol(smiles, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception:
        return None
    return mol


def atom_equal(a1, a2):
    return (
        a1.GetSymbol() == a2.GetSymbol()
        and a1.GetFormalCharge() == a2.GetFormalCharge()
    )


def copy_bond_dir(product: Chem.rdchem.Mol, pre_react: Chem.rdchem.Mol):
    """copy the direction of bonds from the product molecule to the predicted reactant molecules"""
    bond_dir_map = {}
    bond_stereo_map = {}

    for bond in product.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        if bond.GetBondDir() != Chem.rdchem.BondDir.NONE:
            bond_dir_map[(begin_atom, end_atom)] = bond.GetBondDir()

        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            bond_stereo_map[(begin_atom, end_atom)] = bond.GetStereo()

    change_mol = Chem.RWMol(pre_react)
    for bond in change_mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        begin_atom_mapnum = begin_atom.GetAtomMapNum()
        end_atom_mapnum = end_atom.GetAtomMapNum()

        if begin_atom_mapnum == 0 or end_atom_mapnum == 0:
            continue

        if (end_atom_mapnum, begin_atom_mapnum) in bond_stereo_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum

        if (begin_atom_mapnum, end_atom_mapnum) in bond_stereo_map:
            bond.SetStereo(
                bond_stereo_map[(begin_atom_mapnum, end_atom_mapnum)])

        if (end_atom_mapnum, begin_atom_mapnum) in bond_dir_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum

        if (begin_atom_mapnum, end_atom_mapnum) in bond_dir_map:
            bond.SetBondDir(bond_dir_map[(begin_atom_mapnum, end_atom_mapnum)])

    return change_mol


def copy_bond_dir_fwd(src, dest):
    """copy the direction of bonds from the product molecule to the predicted reactant molecules"""
    bond_dir_map = {}
    bond_stereo_map = {}
    bond_type_map = {}

    for bond in src.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        begin_atom, end_atom = (begin_atom, end_atom) if begin_atom < end_atom else (
            end_atom, begin_atom)
        if bond.GetBondDir() != Chem.rdchem.BondDir.NONE:
            bond_dir_map[(begin_atom, end_atom)] = bond.GetBondDir()

        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            bond_stereo_map[(begin_atom, end_atom)] = bond.GetStereo()
            bond_type_map[(begin_atom, end_atom)] = bond.GetBondType()

    dest_bond_type_map = {}
    change_mol = Chem.RWMol(dest)
    for bond in change_mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        begin_atom, end_atom = (begin_atom, end_atom) if begin_atom < end_atom else (
            end_atom, begin_atom)
        dest_bond_type_map[(begin_atom, end_atom)] = bond.GetBondType()

    # # for DEBUG
    # draw_mols([src, dest])
    # print(bond_dir_map)
    # print(bond_stereo_map)
    # print('-' * 50)
    # dest = Chem.RWMol(dest)
    # print(type(change_mol))     # <class 'rdkit.Chem.rdchem.RWMol'>

    drift = 0
    for key in bond_stereo_map:
        begin_atom, end_atom = key
        if bond_type_map[key] == Chem.rdchem.BondType.DOUBLE:
            if dest_bond_type_map.get(key) != bond_type_map[key] and dest_bond_type_map.get((begin_atom - 1, end_atom - 1)) == bond_type_map[key]:
                drift = 1
            if dest_bond_type_map.get(key) != bond_type_map[key] and dest_bond_type_map.get((begin_atom + 1, end_atom + 1)) == bond_type_map[key]:
                drift = -1

    change_mol = Chem.RWMol(dest)
    for bond in change_mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        begin_atom_mapnum = begin_atom.GetAtomMapNum()
        end_atom_mapnum = end_atom.GetAtomMapNum()
        begin_atom_mapnum += drift
        end_atom_mapnum += drift

        if begin_atom_mapnum == 0 or end_atom_mapnum == 0:
            continue

        if (end_atom_mapnum, begin_atom_mapnum) in bond_stereo_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum

        if (begin_atom_mapnum, end_atom_mapnum) in bond_stereo_map:
            bond.SetStereo(
                bond_stereo_map[(begin_atom_mapnum, end_atom_mapnum)])
            # print("bond_stereo_map:", begin_atom_mapnum, end_atom_mapnum)

        if (end_atom_mapnum, begin_atom_mapnum) in bond_dir_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum

        if (begin_atom_mapnum, end_atom_mapnum) in bond_dir_map:
            bond.SetBondDir(bond_dir_map[(begin_atom_mapnum, end_atom_mapnum)])

    bond_dir_map_new = {}
    bond_stereo_map_new = {}
    for bond in change_mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        if bond.GetBondDir() != Chem.rdchem.BondDir.NONE:
            bond_dir_map_new[(begin_atom, end_atom)] = bond.GetBondDir()

        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            bond_stereo_map_new[(begin_atom, end_atom)] = bond.GetStereo()

    # # for DEBUG
    # print(bond_dir_map_new)
    # print(bond_stereo_map_new)
    # draw_mol(change_mol)
    # exit()
    return change_mol


def draw_mol(mol):
    FIG_PATH = './nag2g_NAG2G/NAG2G/outputs/figs/'

    if isinstance(mol, str):
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(mol)
    try:
        # 保存分子结构图为PNG文件
        time_fmt_str = "%Y%m%d-%H%M%S.%f"
        curr_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        # time_str = curr_time
        time_str = curr_time.strftime(time_fmt_str)
        output_file = f"molecule_{time_str}.png"
        filename = os.path.join(FIG_PATH, output_file)
        Chem.Draw.MolToImageFile(mol, filename)
    except:
        print(f"{mol} drawed failed")
    else:
        print(f"{Chem.MolToSmiles(mol, isomericSmiles=True)} saved to \n\t{filename}")


def draw_mols(mols):
    for i, mol in enumerate(mols):
        print(f"[{i}]", end='\t')
        draw_mol(mol)


def get_atom_map(mol):
    return {
        atom.GetAtomMapNum(): atom.GetSymbol().lower()
        for atom in mol.GetAtoms()
    }


def check_atom_map(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    mol_atom_1 = get_atom_map(mol1)
    mol_atom_2 = get_atom_map(mol2)
    if mol_atom_1 == mol_atom_2:
        return 1
    elif sorted(list(mol_atom_1.values())) == sorted(list(mol_atom_2.values())):
        return 2
    elif len(mol_atom_1) == len(mol_atom_2):
        return 3
    return 0


def add_chirality_retro(product, pred_react, DEBUG_MODE=False):
    """copy the atom chirality and bond direction from the product molecule to the predicted reactant molecule"""
    if DEBUG_MODE:
        print(f"product: {product}\npred_react: {pred_react}")

    prod_mol = Chem.MolFromSmiles(product)
    react_mol = Chem.MolFromSmiles(pred_react)
    if DEBUG_MODE:
        draw_mol(product)
        draw_mol(pred_react)

    react_atom_map = {}

    for atom in react_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        react_atom_map[mapnum] = atom

    for atom in prod_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        ratom = react_atom_map[mapnum]
        copy_chirality(atom, ratom)

    if DEBUG_MODE:
        draw_mol(react_mol)
    chiral_react_smiles = Chem.MolToSmiles(react_mol, isomericSmiles=True)
    react_mol = Chem.MolFromSmiles(chiral_react_smiles)
    change_react_mol = copy_bond_dir(prod_mol, react_mol)
    # change_react_mol = Chem.RWMol(react_mol)
    if DEBUG_MODE:
        draw_mol(react_mol)

    SanitizeMol(
        change_react_mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False
    )
    AssignStereochemistry(
        change_react_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )

    chiral_react_smiles = Chem.MolToSmiles(
        change_react_mol, isomericSmiles=True)

    if DEBUG_MODE:
        draw_mol(react_mol)

    return chiral_react_smiles


def add_chirality_fwd(pred_product, reactants, DEBUG_MODE=False):
    """copy the atom chirality and bond direction from the product molecule to the predicted reactant molecule"""
    if DEBUG_MODE:
        print(f"pred_product: {pred_product}\nreactants: {reactants}")

    prod_mol = Chem.MolFromSmiles(pred_product)
    react_mol = Chem.MolFromSmiles(reactants)

    if DEBUG_MODE:
        draw_mol(pred_product)

    react_atom_map = {}

    for atom in react_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        react_atom_map[mapnum] = atom

    for atom in prod_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        try:
            ratom = react_atom_map[mapnum]
        except KeyError:
            pass
        else:
            copy_chirality(ratom, atom)

    if DEBUG_MODE:
        draw_mol(prod_mol)

    chiral_product_smiles = Chem.MolToSmiles(prod_mol, isomericSmiles=True)
    prod_mol = Chem.MolFromSmiles(chiral_product_smiles)

    if DEBUG_MODE:
        draw_mol(prod_mol)

    # TODO
    change_prod_mol = copy_bond_dir_fwd(react_mol, prod_mol)

    if DEBUG_MODE:
        draw_mol(change_prod_mol)

    SanitizeMol(
        change_prod_mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False
    )
    AssignStereochemistry(
        change_prod_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )

    chiral_prod_smiles = Chem.MolToSmiles(change_prod_mol, isomericSmiles=True)

    if DEBUG_MODE:
        draw_mol(chiral_prod_smiles)

    return chiral_prod_smiles


def add_chirality(gt_mol, pred_mol, task='retrosynthesis', DEBUG_MODE=False):
    if task == 'retrosynthesis':
        return add_chirality_retro(gt_mol, pred_mol, DEBUG_MODE)
    elif task == 'synthesis':
        return add_chirality_fwd(pred_mol, gt_mol, DEBUG_MODE)


def is_sim(smile1, smile2):
    try:
        smile1 = canonicalize(smile1)
        smile2 = canonicalize(smile2)
    except:
        return False
    if smile1 == smile2:
        return True
    else:
        return False


if __name__ == "__main__":
    gt_reactant = "O=C1CCC(=O)N1[Br:11].[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])/[CH:8]=[CH:9]/[CH3:10]"
    gt_product = "[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])/[CH:8]=[CH:9]/[CH2:10][Br:11]"
    from mol_graph_basic import get_canonical_smile, same_smi

    MODE = 'retrosynthesis'
    MODE = 'synthesis'
    MODE = 'test'
    MODE = 'debug'
    if MODE == 'retrosynthesis':
        pure_reactant = get_canonical_smile(gt_reactant, isomericSmiles=False)
        pred = add_chirality(gt_product, pure_reactant)
        print(
            f"pure: {same_smi(gt_reactant, pure_reactant)}, chiral: {same_smi(gt_reactant, pred)}")
        draw_mol(pred)
    elif MODE == 'synthesis':
        pure_product = get_canonical_smile(gt_product, isomericSmiles=False)
        pred_product = add_chirality(
            pure_product, gt_reactant, task=MODE, DEBUG_MODE=True)
        print(
            f"pure: {same_smi(gt_product, pure_product)}, chiral: {same_smi(gt_product, pred_product)}")
    elif MODE == 'debug':
        # atom mapping lossed(1, 66), wrong(45, 50, 52, 54, 58, 69)
        # 出现在断键位置(53)
        # 原料中不包含信息(55)
        json_filename = '/data/users/yaolin/NAG2G/outputs/weights_20230717/NAG2G_G2G_uspto_50k_b16_1_l6_vnode1_wda_true_lp_0_0_nsum2_false_sep2_false_cls_false_hdegree_true_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20230717-154639/checkpoint_last/strict_failed_samples.json'
        with open(json_filename, 'r') as fp:
            failed_samples = json.load(fp)

        for i, sample in enumerate(failed_samples):
            # if i < 9:
            #     continue
            _, product, target, pred, *_ = sample.values()
            gt_reactant, (gt_product_str,
                          gt_product_uns), pred_product = product, target, pred

            # # retro
            # pure_reactant = get_canonical_smile(gt_reactant, isomericSmiles=False)
            # ch_reactant = add_chirality(gt_product_str, pure_reactant)
            # print(pure_reactant, ch_reactant, sep='\n')
            # exit()

            # synthesis
            chiral_product = add_chirality(
                pred_product, gt_reactant, task='synthesis', DEBUG_MODE=False)
            draw_mols([gt_reactant, gt_product_str, gt_product_uns,
                      pred_product, chiral_product])

            print("Strict result:", same_smi(gt_product_str, chiral_product))

            exit()

    else:
        # mol1 = "[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])/[CH:8]=[CH:9]/[CH2:10][Br:11] "
        # mol2 = "[Br:8][CH2:18][CH:17]=[CH:16][C:14]([O:13][Si:10]([CH3:9])([CH3:11])[CH3:12])=[O:15]"
        mols = [
            "[CH3:1][CH2:2][C@H:3]([CH3:4])[NH:5][c:6]1[cH:7][c:8]([C:9](=[O:10])[O:11][CH3:12])[cH:13][c:14]([CH:15]([F:16])[F:17])[n:18]1 ",
            "[c:2]1([NH:19][CH:17]([CH2:16][CH3:15])[CH3:18])[cH:3][c:4]([C:5](=[O:6])[O:7][CH3:8])[cH:9][c:10]([CH:11]([F:12])[F:13])[n:14]1"
        ]

        draw_mols(mols)
        exit()

        mol1 = '[CH3:1][C:2](=[O:3])[c:4]1[n:5][c:6]2[cH:7][c:8]([NH:9][C:10](=[O:11])[c:12]3[cH:13][cH:14][c:15](/[CH:16]=[CH:17]/[C:18]([F:19])([F:20])[F:21])[cH:22][c:23]3[CH3:24])[cH:25][cH:26][c:27]2[s:28]1 '
        mol2 = '[H][C:1]([H])([H])[C:2](=[O:3])[c:4]1[n:5][c:6]2[c:7]([H])[c:8]([N:9]([H])[C:10](=[O:11])[c:12]3[c:13]([H])[c:14]([H])[c:15]([C:16]([H])=[C:17]([H])[C:18]([F:19])([F:20])[F:21])[c:22]([H])[c:23]3[C:24]([H])([H])[H])[c:25]([H])[c:26]([H])[c:27]2[s:28]1'

        check_atom_map(mol1, mol2)

        exit()
        # CC(C)(C)OC(=O)O[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18].[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][nH:11]2)[cH:19]1 gt_product
        temp = 'C(=[O:30])([CH:31]([NH:1][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]([S:9](=[O:10])(=[O:11])[NH:12][c:13]2[cH:14][cH:15][cH:16][cH:17][c:18]2[O:19][c:20]2[cH:21][cH:22][c:23]([Cl:24])[cH:25][c:26]2[Cl:27])[cH:28][cH:29]1)[CH3:2])[NH:32][CH2:33][CH:34]1[CH2:35][CH2:36][N:37]([C:38](=[O:39])[O:40][C:41]([CH3:42])([CH3:43])[CH3:44])[CH2:45][CH2:46]1'
        draw_mol(temp)
        exit()

        temps = [
            r'C\\C=C/Cl',
            r'C\\C=C\\Cl',
            r'C/C=C/Cl',
            r'CC=CCl',
        ]
        draw_mols(temps)
        for i in range(4):
            for j in range(4):
                print(same_smi(temps[i], temps[j]), end='\t')
            print()
        exit()

        temp = r'C\\C=C/Cl'
        draw_mol(temp)
        exit()

        temp_smi = '[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])/[CH:8]=[CH:9]/[CH3:10]'

        temp_smi1 = '[H]C([H])([H])C(OC(=O)[n:3]1[c:2]([H])[c:1]([H])[c:18]2[c:4]([H])[c:14]([C:12](C([H])([H])[H])=[O:13])[c:15]([H])[c:16]([H])[c:17]12)(C([H])([H])[H])C([H])([H])[H]'
        temp_smi2 = '[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][n:11]2[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18])[cH:19]1 '
        # temp1 = get_canonical_smile(temp_smi1, isomericSmiles=False)
        # temp2 = get_canonical_smile(temp_smi2, isomericSmiles=False)
        print(same_smi(temp_smi1, temp_smi2))
