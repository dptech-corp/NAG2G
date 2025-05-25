from rdkit import Chem
import random

try:
    from rdkit.Chem import Draw

    def draw_mol(smis, save_path, mols_per_row=4, img_size=(400, 400)):
        mols = []
        for smi in smis:
            try:
                mol = Chem.MolFromSmiles(smi)
            except:
                mol = None
            mols.append(mol)
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=img_size,
            legends=["" for x in mols],
        )
        img.save(save_path)

except:
    pass


def get_InchiKey(smi):
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return None
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
        return key
    except:
        return None


def judge_InchiKey(key1, key2):
    if key1 is None or key2 is None:
        return False
    return key1 == key2


def same_smi(smi1, smi2):
    key1 = get_InchiKey(smi1)
    if key1 is None:
        return False
    key2 = get_InchiKey(smi2)
    if key2 is None:
        return False
    return judge_InchiKey(key1, key2)


def find_invert(nbrs):
    clockwise_orders = {"A": "BCDB", "B": "ADCA", "C": "ABDA", "D": "ACBA"}
    char_list = ["A", "B", "C", "D"]
    nbrs_tmp = {i: char_list[idx] for idx, i in enumerate(nbrs)}
    s_nbrs_tmp = [nbrs_tmp[i] for i in sorted(nbrs)]
    current_order = clockwise_orders[s_nbrs_tmp[0]]
    if (s_nbrs_tmp[1] + s_nbrs_tmp[2]) in current_order:
        return False
    return True


def update_all_atom_stereo(atom):
    chirality = atom.GetChiralTag()
    if chirality in (
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    ):
        nbrs = [x.GetOtherAtomIdx(atom.GetIdx()) for x in atom.GetBonds()]
        invert = find_invert(nbrs)
        if invert:
            if chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                return Chem.ChiralType.CHI_TETRAHEDRAL_CW
            elif chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                return Chem.ChiralType.CHI_TETRAHEDRAL_CCW

    return chirality


def extract_info_from_smiles(smiles):
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if False:
        list_ = [i for i, _ in enumerate(mol.GetAtoms())]
        random.shuffle(list_)
        mol = Chem.RenumberAtoms(mol, list_)
    atom_info = []
    for atom in mol.GetAtoms():
        atom_info.append(
            {
                "symbol": atom.GetSymbol(),
                "charge": atom.GetFormalCharge(),
                "hydrogens": atom.GetNumExplicitHs(),
                # "hydrogens_i": atom.GetNumImplicitHs(),
                "chirality": update_all_atom_stereo(atom),
                "n_radical_electrons": atom.GetNumRadicalElectrons(),
                "atommap": atom.GetAtomMapNum(),
            }
        )
    bond_info = []
    for bond in mol.GetBonds():
        bond_info.append(
            {
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "bond_type": bond.GetBondType(),
                "stereo": bond.GetStereo(),
                "bond_dir": bond.GetBondDir(),
            }
        )
    # random.shuffle(bond_info)

    return atom_info, bond_info


def reconstruct_smiles(atom_info, bond_info):
    mol = Chem.RWMol()
    atom_idx_map = {}
    for idx, info in enumerate(atom_info):
        if info["symbol"] == "NoneAtom":
            continue
        atom = Chem.Atom(info["symbol"])
        atom.SetNumExplicitHs(info["hydrogens"])
        atom.SetNumRadicalElectrons(info["n_radical_electrons"])
        # if info["hydrogens_i"] == 0:
        #     atom.SetNoImplicit(True)
        atom.SetFormalCharge(info["charge"])
        atom.SetChiralTag(info["chirality"])
        atom.SetAtomMapNum(info["atommap"])
        atom_idx = mol.AddAtom(atom)
        atom_idx_map[idx] = atom_idx

    for info in bond_info:
        if info["begin_atom"] > info["end_atom"]:
            info["begin_atom"], info["end_atom"] = info["end_atom"], info["begin_atom"]
    bond_info = sorted(bond_info, key=lambda x: (x["begin_atom"], x["end_atom"]))
    for info in bond_info:
        mol.AddBond(
            atom_idx_map[info["begin_atom"]],
            atom_idx_map[info["end_atom"]],
            info["bond_type"],
        )
        bond = mol.GetBondBetweenAtoms(
            atom_idx_map[info["begin_atom"]], atom_idx_map[info["end_atom"]]
        )
        bond.SetStereo(info["stereo"])
        bond.SetBondDir(info["bond_dir"])
    mol = mol.GetMol()
    try:
        mol.UpdatePropertyCache()
    except:
        pass
    # Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles


def test(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if True:
        # [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        [a.SetIsotope(0) for a in mol.GetAtoms()]
        smiles_refined = Chem.MolToSmiles(mol, doRandom=True)
    else:
        smiles_refined = smiles

    atom_info, bond_info = extract_info_from_smiles(smiles_refined)
    # print(atom_info)
    # print(bond_info)
    reconstructed_smiles = reconstruct_smiles(atom_info, bond_info)
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


def check_all():
    smiles = [
        "O=C(OCc1ccccc1)[NH:10][CH2:9][CH2:8][CH2:7][CH2:6][C@@H:5]([C:3]([O:2][CH3:1])=[O:4])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]"
    ]

    import pandas as pd
    from tqdm import tqdm

    p = "./USPTO50K_raw_20230220/test.csv"
    df = pd.read_csv(p)
    smiles = list(df["rxn_smiles"])
    smiles = [i.split(">")[0] for i in smiles]
    flag = 0
    for smi in tqdm(smiles):
        if not test(smi):
            flag += 1
    print(flag)


if __name__ == "__main__":
    check_all()
