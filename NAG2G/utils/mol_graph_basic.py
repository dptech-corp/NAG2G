from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from .features_dict2 import bond_type_list, chiral_type_list_1, chiral_type_list, bond_stereo_list
try:
    from rdkit.Chem import Draw
except:
    print("can not import chem draw")
from itertools import product
from copy import deepcopy
from collections import OrderedDict

# from basic import draw_mol

np.set_printoptions(threshold=np.inf)

flag_kekulize = False
flag_atoms_chiraltag = "new"


def draw_mol(smis, save_path, mols_per_row=4, img_size=(400, 400)):
    mols = []
    for smi in smis:
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            mol = None
        mols.append(mol)
    img = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=img_size, legends=["" for x in mols]
    )
    img.save(save_path)


def get_adjacency_matrix(smiles, add_h=None):
    mol = Chem.MolFromSmiles(smiles)
    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    if add_h is True:
        mol = AllChem.AddHs(mol)
    elif add_h is False:
        mol = AllChem.RemoveHs(mol)

    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    bond_stereo = np.zeros_like(adjacency_matrix)
    bond_stereo_dict = dict()

    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_type_list.index(bond.GetBondType())
        adjacency_matrix[begin_idx, end_idx] = bond_type
        adjacency_matrix[end_idx, begin_idx] = bond_type

        bond_stereo_value = bond.GetStereo()
        bond_stereo[begin_idx, end_idx] = bond_stereo_value
        bond_stereo[end_idx, begin_idx] = bond_stereo_value

        stereo_atoms = list(bond.GetStereoAtoms())
        if len(stereo_atoms) >= 2:
            bond_stereo_dict[(begin_idx, end_idx)] = stereo_atoms
            bond_stereo_dict[(end_idx, begin_idx)] = stereo_atoms

    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    atoms_charge = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_h_number = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]

    if flag_atoms_chiraltag == "old":
        atoms_chiraltag = [
            chiral_type_list_1.index(atom.GetChiralTag()) for atom in mol.GetAtoms()
        ]
    else:
        atoms_chiraltag = [0 for _ in range(mol.GetNumAtoms())]
        for i, c in Chem.FindMolChiralCenters(mol):
            atoms_chiraltag[i] = chiral_type_list.index(c)

    return {
        "adjacency_matrix": adjacency_matrix,
        "atoms": atoms,
        "atoms_map": atoms_map,
        "atoms_chiraltag": atoms_chiraltag,
        "atoms_charge": atoms_charge,
        "bond_stereo": bond_stereo,
        "bond_stereo_dict": bond_stereo_dict,
        "atom_h_number": atom_h_number,
    }


def create_molecule_with_atoms(
    atoms, atoms_map, atoms_charge, atoms_chiraltag, atom_h_number
):
    molecule = Chem.RWMol()
    atom_index = []

    for atom_number, atom_symbol in enumerate(atoms):
        atom_tmp = Chem.Atom(atom_symbol)
        if atoms_map is not None:
            atom_tmp.SetAtomMapNum(atoms_map[atom_number])
        if atoms_charge is not None:
            atom_tmp.SetFormalCharge(atoms_charge[atom_number])
        if atoms_chiraltag is not None and flag_atoms_chiraltag == "old":
            atom_tmp.SetChiralTag(chiral_type_list_1[atoms_chiraltag[atom_number]])
        if atom_h_number is not None:
            atom_tmp.SetNumExplicitHs(atom_h_number[atom_number])

        molecular_index = molecule.AddAtom(atom_tmp)
        atom_index.append(molecular_index)

    return molecule, atom_index


def add_bonds_to_molecule(molecule, atom_index, adjacency_matrix, bond_type_list):
    for index_x, row_vector in enumerate(adjacency_matrix):
        for index_y, bond in enumerate(row_vector[index_x + 1 :], start=index_x + 1):
            if bond != 0:
                molecule.AddBond(
                    atom_index[index_x], atom_index[index_y], bond_type_list[bond]
                )


def set_bond_stereo(molecule, bond_stereo, bond_stereo_list, bond_stereo_dict):
    for bond in molecule.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        stereo_tmp = bond_stereo[begin_idx, end_idx]
        bond.SetStereo(bond_stereo_list[stereo_tmp])

        if (begin_idx, end_idx) in bond_stereo_dict.keys():
            stereo_atoms = bond_stereo_dict[(begin_idx, end_idx)]
            bond.SetStereoAtoms(stereo_atoms[0], stereo_atoms[1])


def update_molecule_property_cache(molecule):
    try:
        molecule.UpdatePropertyCache()
    except:
        pass


def assign_chiral_tags(molecule, atoms_chiraltag, atom_index, chiral_type_list):
    trials = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    ]
    chis = OrderedDict()
    for i, c in enumerate(atoms_chiraltag):
        if c > 0:
            chis.update({atom_index[i]: chiral_type_list[c]})

    if chis:
        for prod in product(trials, repeat=len(chis)):
            m = deepcopy(molecule)
            for atIdx, chiral_tag in zip(chis.keys(), prod):
                m.GetAtomWithIdx(atIdx).SetChiralTag(chiral_tag)

            Chem.AssignStereochemistry(m)
            matches = [chis[atIdx] == c for atIdx, c in Chem.FindMolChiralCenters(m)]
            if all(matches):
                molecule = m
    else:
        Chem.AssignStereochemistry(molecule)
    return molecule


def get_molecule_smiles(molecule, flag_kekulize, add_h):
    # Chem.AssignAtomChiralTagsFromStructure(molecule)
    # molecule = AllChem.RemoveHs(molecule)
    # return molecule
    if flag_kekulize:
        smiles = Chem.MolToSmiles(molecule, kekuleSmiles=True)
    else:
        if add_h:
            molecule = AllChem.AddHs(molecule)
        # else:
        # molecule = AllChem.RemoveHs(molecule)
        smiles = Chem.MolToSmiles(molecule)
    return smiles


def graph2mol(
    adjacency_matrix,
    atoms,
    atoms_map=None,
    atoms_chiraltag=None,
    atoms_charge=None,
    bond_stereo=None,
    bond_stereo_dict=None,
    atom_h_number=None,
    add_h=False,
):
    molecule, atom_index = create_molecule_with_atoms(
        atoms, atoms_map, atoms_charge, atoms_chiraltag, atom_h_number
    )

    add_bonds_to_molecule(molecule, atom_index, adjacency_matrix, bond_type_list)

    if bond_stereo is not None:
        set_bond_stereo(molecule, bond_stereo, bond_stereo_list, bond_stereo_dict)

    # if atom_h_number is not None:
    #     set_h_number(molecule, atom_h_number)

    molecule = molecule.GetMol()

    update_molecule_property_cache(molecule)

    if atoms_chiraltag is not None and flag_atoms_chiraltag == "new":
        molecule = assign_chiral_tags(
            molecule, atoms_chiraltag, atom_index, chiral_type_list
        )

    smiles = get_molecule_smiles(molecule, flag_kekulize, add_h)

    return smiles


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


def get_charge_dict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    charge_dict = {
        atom.GetAtomMapNum(): atom.GetFormalCharge()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() != 0
    }
    return dict(sorted(charge_dict.items()))


def get_dict(path):
    with open(path, "r") as f:
        a = [i.strip() for i in f.readlines()]
    return a


def get_canonical_smile(testsmi, isomericSmiles=True):
    try:
        mol = Chem.MolFromSmiles(testsmi)
        return Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    except:
        print("Cannot convert {} to canonical smiles")
        return testsmi


def error(testsmi):
    try:
        mol = Chem.MolFromSmiles(testsmi)
        _ = Chem.MolToSmiles(mol)
        return True
    except:
        return False


def drop_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def setmap2smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.RemoveHs(mol)
    [atom.SetAtomMapNum(idx + 1) for idx, atom in enumerate(mol.GetAtoms())]
    return Chem.MolToSmiles(mol)


def test(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.RemoveHs(mol)
    # chis1 = list(Chem.FindMolChiralCenters(mol))
    # print([bo.GetStereo() for bo in mol.GetBonds()])

    # Draw.MolToFile(mol, "test1.png", (1000, 1000))

    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        smiles_refined = Chem.MolToSmiles(mol, kekuleSmiles=True)
    else:
        if True:
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            smiles_refined = Chem.MolToSmiles(mol)
        else:
            smiles_refined = smiles
    kwargs = get_adjacency_matrix(smiles_refined, add_h=None)

    smiles2 = graph2mol(**kwargs)
    # chis2 = list(Chem.FindMolChiralCenters(mol))
    # print([tuple(bo.GetStereoAtoms()) for bo in mol.GetBonds()])
    # Draw.MolToFile(mol, "test2.png", (1000, 1000))
    same = same_smi(smiles_refined, smiles2)
    if not same:
        print("*" * 10)
        print(smiles)
        print(smiles_refined)
        print(smiles2)
        # print(chis1, chis2)
        draw_mol([smiles, smiles_refined, smiles2], "2.png")
        # raise
    return same


def test2(smiles, lis):
    mol = Chem.MolFromSmiles(smiles)
    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for i in atoms:
        if i not in lis:
            print(i)
            return False
    return True


if __name__ == "__main__":
    smiles = [
        # "[CH3:1][CH2:2][C:3](=[O:4])[O:5][C@H:6]1[CH2:7][CH2:8][C@H:9]2[C@@H:10]3[CH2:11][CH2:12][C:13]4=[CH:14][C:15](=[O:16])[CH2:17][CH2:18][C@:19]4([CH2:20][OH:21])[C@H:22]3[CH2:23][CH2:24][C@:25]12[CH3:26]",
        # "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH:6]([CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH:12]=[CH:13][CH2:14][C@H:15]1[c:16]2[cH:17][cH:18][c:19]([O:20][CH3:21])[cH:22][c:23]2[S:24][CH2:25][C@@:26]1([CH3:27])[c:28]1[cH:29][cH:30][c:31]([O:32][CH3:33])[cH:34][cH:35]1)[CH2:36][CH2:37][CH2:38][C:39]([F:40])([F:41])[C:42]([F:43])([F:44])[F:45]",
        # "[O:1]=[C:2]([OH:3])[CH2:4][C@@H:5]1[CH2:6][c:7]2[cH:8][c:9]([Br:10])[c:11]3[nH:12][n:13][c:14]([Cl:15])[c:16]3[c:17]2[CH2:18][N:19]([CH2:20][C:21]([F:22])([F:23])[F:24])[C:25]1=[O:26]",
        "O=C(OCc1ccccc1)[NH:10][CH2:9][CH2:8][CH2:7][CH2:6][C@@H:5]([C:3]([O:2][CH3:1])=[O:4])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]"
    ]

    import pandas as pd
    from tqdm import tqdm

    # a = get_dict(
    #     "/data/users/yaolin/NAG2G_/retro/USPTO50K_ALL_20221216_1_rmh/encoder_dict.txt"
    # )
    # df = pd.read_csv("/data/users/yaolin/NAG2G_/USPTO50K_raw_20230220/train.csv")
    # smiles = list(df["rxn_smiles"])
    # smiles = [i.split('>')[0] for i in smiles]
    flag = 0
    for smi in tqdm(smiles):
        if not test(smi):
            flag += 1
    print(flag)
