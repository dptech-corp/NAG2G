from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

try:
    from rdkit.Chem import Draw
except:
    print("can not import chem draw")
from itertools import product
from copy import deepcopy
from collections import OrderedDict


np.set_printoptions(threshold=np.inf)

flag_kekulize = False
flag_atoms_chiraltag = "new"
flag_use_list = False

# 22 type
bond_type_list = [
    Chem.rdchem.BondType.UNSPECIFIED,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.QUADRUPLE,
    Chem.rdchem.BondType.QUINTUPLE,
    Chem.rdchem.BondType.HEXTUPLE,
    Chem.rdchem.BondType.ONEANDAHALF,
    Chem.rdchem.BondType.TWOANDAHALF,
    Chem.rdchem.BondType.THREEANDAHALF,
    Chem.rdchem.BondType.FOURANDAHALF,
    Chem.rdchem.BondType.FIVEANDAHALF,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.IONIC,
    Chem.rdchem.BondType.HYDROGEN,
    Chem.rdchem.BondType.THREECENTER,
    Chem.rdchem.BondType.DATIVEONE,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.DATIVEL,
    Chem.rdchem.BondType.DATIVER,
    Chem.rdchem.BondType.OTHER,
    Chem.rdchem.BondType.ZERO,
]

chiral_type_list_1 = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,  # chirality that hasn't been specified
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # tetrahedral: clockwise rotation (SMILES @@)
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,  # tetrahedral: counter-clockwise rotation (SMILES @)
    Chem.rdchem.ChiralType.CHI_OTHER,  # some unrecognized type of chirality
    # Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,  # tetrahedral, use permutation flag
    # Chem.rdchem.ChiralType.CHI_ALLENE,  # allene, use permutation flag
    # Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,  # square planar, use permutation flag
    # Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,  # trigonal bipyramidal, use permutation flag
    # Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,  # octahedral, use permutation flag
]

chiral_type_list = ["", "S", "R"]

bond_stereo_list = [  # stereochemistry of double bonds
    Chem.rdchem.BondStereo.STEREONONE,  # no special style
    Chem.rdchem.BondStereo.STEREOANY,  # intentionally unspecified
    # -- Put any true specifications about this point so
    # that we can do comparisons like if(bond->getStereo()>Bond::STEREOANY)
    Chem.rdchem.BondStereo.STEREOZ,  # Z double bond
    Chem.rdchem.BondStereo.STEREOE,  # E double bond
    Chem.rdchem.BondStereo.STEREOCIS,  # cis double bond
    Chem.rdchem.BondStereo.STEREOTRANS,  # trans double bond
]


def set_h_number(mol, atom_h_number):
    for i in range(len(atom_h_number)):
        for _ in range(atom_h_number[i]):
            atom_tmp = Chem.Atom("H")
            molecular_index = mol.AddAtom(atom_tmp)
            try:
                mol.AddBond(i, molecular_index, Chem.rdchem.BondType.SINGLE)
            except:
                mol.RemoveAtom(molecular_index)


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


def create_molecule_with_atoms(atoms, atoms_map, atoms_charge, atoms_chiraltag):
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


def assign_chiral_tags(
    molecule, atoms_chiraltag, atom_index, chiral_type_list, flag_use_list
):
    trials = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    ]
    chis = OrderedDict()
    for i, c in enumerate(atoms_chiraltag):
        if c > 0:
            chis.update({atom_index[i]: chiral_type_list[c]})

    if chis:
        if flag_use_list:
            molecule_list = []
        for prod in product(trials, repeat=len(chis)):
            m = deepcopy(molecule)
            for atIdx, chiral_tag in zip(chis.keys(), prod):
                m.GetAtomWithIdx(atIdx).SetChiralTag(chiral_tag)

            Chem.AssignStereochemistry(m)
            matches = [chis[atIdx] == c for atIdx, c in Chem.FindMolChiralCenters(m)]
            if all(matches):
                if flag_use_list:
                    molecule_list.append(m)
                else:
                    molecule = m
                    break
        if flag_use_list:
            molecule = molecule_list
    else:
        Chem.AssignStereochemistry(molecule)
        if flag_use_list:
            molecule = [molecule]
    return molecule


def get_molecule_smiles(molecule, flag_kekulize, flag_use_list, add_h):
    # Chem.AssignAtomChiralTagsFromStructure(molecule)
    # molecule = AllChem.RemoveHs(molecule)
    # return molecule
    if flag_kekulize:
        smiles = Chem.MolToSmiles(molecule, kekuleSmiles=True)
    else:
        # molecule = AllChem.RemoveHs(molecule)
        if flag_use_list:
            # smiles = [Chem.MolToSmiles(AllChem.RemoveHs(i)) for i in molecule]
            if isinstance(molecule, list):  # ?
                molecule = [molecule]
            if add_h:
                molecule = [AllChem.AddHs(i) for i in molecule]
            smiles = [Chem.MolToSmiles(i) for i in molecule]
        else:
            # molecule = AllChem.RemoveHs(molecule)
            if add_h:
                molecule = AllChem.AddHs(molecule)
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
        atoms, atoms_map, atoms_charge, atoms_chiraltag
    )

    add_bonds_to_molecule(molecule, atom_index, adjacency_matrix, bond_type_list)

    if bond_stereo is not None:
        set_bond_stereo(molecule, bond_stereo, bond_stereo_list, bond_stereo_dict)

    if atom_h_number is not None:
        set_h_number(molecule, atom_h_number)

    molecule = molecule.GetMol()

    update_molecule_property_cache(molecule)

    if atoms_chiraltag is not None and flag_atoms_chiraltag == "new":
        molecule = assign_chiral_tags(
            molecule, atoms_chiraltag, atom_index, chiral_type_list, flag_use_list
        )

    smiles = get_molecule_smiles(molecule, flag_kekulize, flag_use_list, add_h)

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
        if False:
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            smiles_refined = Chem.MolToSmiles(mol)
        else:
            smiles_refined = smiles
    kwargs = get_adjacency_matrix(smiles_refined, add_h=None)

    smiles2 = graph2mol(**kwargs)
    # chis2 = list(Chem.FindMolChiralCenters(mol))
    # print([tuple(bo.GetStereoAtoms()) for bo in mol.GetBonds()])
    # Draw.MolToFile(mol, "test2.png", (1000, 1000))
    if flag_use_list:
        same = [same_smi(smiles_refined, i) for i in smiles2]
        same = True if True in same else False
    else:
        same = same_smi(smiles_refined, smiles2)
    if not same:
        print("*" * 10)
        print(smiles)
        print(smiles_refined)
        print(smiles2)
        # print(chis1, chis2)
    # draw_mol([smiles, smiles_refined, smiles2], "2.png")
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