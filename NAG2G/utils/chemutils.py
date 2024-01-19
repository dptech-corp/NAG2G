# Modified from https://github.com/wengong-jin/iclr19-graph2graph
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


def copy_bond_dir(product, pre_react):
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
            bond.SetStereo(bond_stereo_map[(begin_atom_mapnum, end_atom_mapnum)])

        if (end_atom_mapnum, begin_atom_mapnum) in bond_dir_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum

        if (begin_atom_mapnum, end_atom_mapnum) in bond_dir_map:
            bond.SetBondDir(bond_dir_map[(begin_atom_mapnum, end_atom_mapnum)])

    return change_mol


def add_chirality(product, pred_react):
    """copy the atom chirality and bond direction from the product molecule to the predicted reactant molecule"""
    prod_mol = Chem.MolFromSmiles(product)
    react_mol = Chem.MolFromSmiles(pred_react)

    react_atom_map = {}

    for atom in react_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        react_atom_map[mapnum] = atom

    for atom in prod_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()

        ratom = react_atom_map[mapnum]

        copy_chirality(atom, ratom)

    chiral_react_smiles = Chem.MolToSmiles(react_mol, isomericSmiles=True)
    react_mol = Chem.MolFromSmiles(chiral_react_smiles)
    change_react_mol = copy_bond_dir(prod_mol, react_mol)

    SanitizeMol(
        change_react_mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False
    )
    AssignStereochemistry(
        change_react_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )

    chiral_react_smiles = Chem.MolToSmiles(change_react_mol, isomericSmiles=True)

    return chiral_react_smiles


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

    fake_reactant = get_canonical_smile(gt_reactant, isomericSmiles=False)
    pred = add_chirality(gt_product, fake_reactant)
    print(same_smi(gt_reactant, pred))
    print(same_smi(gt_reactant, fake_reactant))
