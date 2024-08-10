import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def get_canonical_smile(testsmi, isomericSmiles=True):
    if testsmi == "":
        return testsmi
    try:
        mol = Chem.MolFromSmiles(testsmi)
        canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    except:
        canonical_smi = testsmi
    return canonical_smi


def get_target_order(smiles_target, check=False, add_h=True):
    mol = Chem.MolFromSmiles(smiles_target)
    if add_h:
        mol = AllChem.AddHs(mol)
    atoms = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    assert (not check) or (0 not in atoms)
    return atoms


def get_atoms(smi, add_h=True):
    mol = Chem.MolFromSmiles(smi)
    if add_h:
        mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    if not add_h:
        atoms = [i for i in atoms if i != "H"]
    return atoms


def csv_file_read(path, usecols=None):
    head_row = pd.read_csv(path, nrows=0)
    print(list(head_row))
    head_row_list = list(head_row)
    if usecols is None:
        usecols = head_row_list
    csv_result = pd.read_csv(path, usecols=usecols)
    row_list = csv_result.values.tolist()
    return row_list


def rm_h_coordinates_map(target_atoms, target_coordinates, target_map):
    assert (
        len(target_atoms) == len(target_map)
        and len(target_atoms) == target_coordinates.shape[1]
    )
    target_atoms_tmp = [i for i in target_atoms if i != "H"]
    idx = [i != "H" for i in target_atoms]
    target_coordinates_tmp = target_coordinates[:, idx]
    target_map_tmp = [
        target_map[i] for i in range(len(target_atoms)) if target_atoms[i] != "H"
    ]
    assert len(target_atoms_tmp) == len(target_map_tmp) and len(target_atoms_tmp) == (
        target_coordinates_tmp.shape[1]
    )
    return target_atoms_tmp, target_coordinates_tmp, target_map_tmp


def renumber_atom_maps(smi):
    if smi == "":
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    atom_map_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(
        num == 0
        for atom, num in zip(mol.GetAtoms(), atom_map_nums)
        if atom.GetAtomicNum() > 1
    ):
        current_map_num = 1
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:  # Heavy atom
                atom.SetAtomMapNum(current_map_num)
                current_map_num += 1
            else:  # H atom
                atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)
