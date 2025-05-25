import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from unicore.data import BaseWrapperDataset
from functools import lru_cache
import warnings

warnings.filterwarnings(action="ignore")


def get_torsions(m):
    m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


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


def inner_smi2coords(smi, num_confs=1, seed=42):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol, addCoords=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

    wt = Descriptors.ExactMolWt(mol)

    if wt > 2000 or len(atoms) < 2:
        print("skip for heavy molecules, and molecules having less than two atoms")
        return None

    res = AllChem.EmbedMolecule(mol, randomSeed=seed)

    if res == 0:
        rotable_bonds = get_torsions(mol)
        for i in range(num_confs):
            np.random.seed(i)
            values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
            for idx in range(len(rotable_bonds)):
                SetDihedral(mol.GetConformer(), rotable_bonds[idx], values[idx])
            Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer())
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = smi2_2Dcoords(smi)

    elif res == -1:
        mol_tmp = Chem.MolFromSmiles(smi)
        AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
        rotable_bonds = get_torsions(mol_tmp)
        for i in range(num_confs):
            np.random.seed(i)
            values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
            for idx in range(len(rotable_bonds)):
                SetDihedral(mol_tmp.GetConformer(), rotable_bonds[idx], values[idx])
            Chem.rdMolTransforms.CanonicalizeConformer(mol_tmp.GetConformer())
            try:
                AllChem.MMFFOptimizeMolecule(mol_tmp)
                coordinates = mol_tmp.GetConformer().GetPositions().astype(np.float32)

            except:
                coordinates = smi2_2Dcoords(smi)
    else:
        coordinates = smi2_2Dcoords(smi)
    atoms, coordinates, atom_map = rm_h_coordinates_map(atoms, np.expand_dims(coordinates, axis=0), atom_map)
    return {"atoms": atoms, "coordinates": coordinates[0], "target_map": atom_map}


class CoordDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
    ):
        super().__init__(dataset)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        smi = self.dataset[index]
        result = inner_smi2coords(smi, num_confs=1, seed=42)
        return result


def setmap2smiles(smiles, check=True):
    mol = Chem.MolFromSmiles(smiles)
    if check:
        atom_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        if sum(atom_map) != 0:
            return smiles
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    mol = AllChem.RemoveHs(mol)
    [atom.SetAtomMapNum(idx + 1) for idx, atom in enumerate(mol.GetAtoms())]
    return Chem.MolToSmiles(mol)


class AddMapDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
    ):
        super().__init__(dataset)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        smiles = self.dataset[index]
        smiles = setmap2smiles(smiles)
        return smiles
