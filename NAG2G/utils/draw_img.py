from rdkit import Chem
try:
    from rdkit.Chem import Draw
except:
    print("can not import chem draw")


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
