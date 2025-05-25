from rdkit import Chem


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
