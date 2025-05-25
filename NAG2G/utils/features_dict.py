from rdkit import Chem


bond_dir_r_dict = {
    "NONE": "NONE",
    "BEGINWEDGE": "R_BEGINWEDGE",
    "BEGINDASH": "R_BEGINDASH",
    "ENDDOWNRIGHT": "R_ENDDOWNRIGHT",
    "ENDUPRIGHT": "R_ENDUPRIGHT",
    "EITHERDOUBLE": "R_EITHERDOUBLE",
    "R_BEGINWEDGE": "BEGINWEDGE",
    "R_BEGINDASH": "BEGINDASH",
    "R_ENDDOWNRIGHT": "ENDDOWNRIGHT",
    "R_ENDUPRIGHT": "ENDUPRIGHT",
    "R_EITHERDOUBLE": "EITHERDOUBLE",
    "UNKNOWN": "UNKNOWN",
}


chirality_dict = {
    "CHI_UNSPECIFIED": Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    "CHI_TETRAHEDRAL_CW": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    "CHI_TETRAHEDRAL_CCW": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    "CHI_TRIGONALBIPYRAMIDAL": Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    "CHI_OCTAHEDRAL": Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
    "CHI_SQUAREPLANAR": Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
    "CHI_OTHER": Chem.rdchem.ChiralType.CHI_OTHER,
    "CHI_TETRAHEDRAL": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
    "CHI_ALLENE": Chem.rdchem.ChiralType.CHI_ALLENE,
}


bond_type_dict = {
    "SINGLE": Chem.rdchem.BondType.SINGLE,
    "DOUBLE": Chem.rdchem.BondType.DOUBLE,
    "TRIPLE": Chem.rdchem.BondType.TRIPLE,
    "AROMATIC": Chem.rdchem.BondType.AROMATIC,
    "UNSPECIFIED": Chem.rdchem.BondType.UNSPECIFIED,
    "QUADRUPLE": Chem.rdchem.BondType.QUADRUPLE,
    "QUINTUPLE": Chem.rdchem.BondType.QUINTUPLE,
    "HEXTUPLE": Chem.rdchem.BondType.HEXTUPLE,
    "ONEANDAHALF": Chem.rdchem.BondType.ONEANDAHALF,
    "TWOANDAHALF": Chem.rdchem.BondType.TWOANDAHALF,
    "THREEANDAHALF": Chem.rdchem.BondType.THREEANDAHALF,
    "FOURANDAHALF": Chem.rdchem.BondType.FOURANDAHALF,
    "FIVEANDAHALF": Chem.rdchem.BondType.FIVEANDAHALF,
    "IONIC": Chem.rdchem.BondType.IONIC,
    "HYDROGEN": Chem.rdchem.BondType.HYDROGEN,
    "THREECENTER": Chem.rdchem.BondType.THREECENTER,
    "DATIVEONE": Chem.rdchem.BondType.DATIVEONE,
    "DATIVE": Chem.rdchem.BondType.DATIVE,
    "DATIVEL": Chem.rdchem.BondType.DATIVEL,
    "DATIVER": Chem.rdchem.BondType.DATIVER,
    "OTHER": Chem.rdchem.BondType.OTHER,
    "ZERO": Chem.rdchem.BondType.ZERO,
}


bond_stereo_dict = {
    "STEREONONE": Chem.rdchem.BondStereo.STEREONONE,
    "STEREOZ": Chem.rdchem.BondStereo.STEREOZ,
    "STEREOE": Chem.rdchem.BondStereo.STEREOE,
    "STEREOCIS": Chem.rdchem.BondStereo.STEREOCIS,
    "STEREOTRANS": Chem.rdchem.BondStereo.STEREOTRANS,
    "STEREOANY": Chem.rdchem.BondStereo.STEREOANY,
}

bond_dir_dict = {
    "NONE": Chem.rdchem.BondDir.NONE,
    "BEGINWEDGE": Chem.rdchem.BondDir.BEGINWEDGE,
    "BEGINDASH": Chem.rdchem.BondDir.BEGINDASH,
    "ENDDOWNRIGHT": Chem.rdchem.BondDir.ENDDOWNRIGHT,
    "ENDUPRIGHT": Chem.rdchem.BondDir.ENDUPRIGHT,
    "EITHERDOUBLE": Chem.rdchem.BondDir.EITHERDOUBLE,
    "R_BEGINWEDGE": Chem.rdchem.BondDir.BEGINWEDGE,
    "R_BEGINDASH": Chem.rdchem.BondDir.BEGINDASH,
    "R_ENDDOWNRIGHT": Chem.rdchem.BondDir.ENDDOWNRIGHT,
    "R_ENDUPRIGHT": Chem.rdchem.BondDir.ENDUPRIGHT,
    "R_EITHERDOUBLE": Chem.rdchem.BondDir.EITHERDOUBLE,
    "UNKNOWN": Chem.rdchem.BondDir.UNKNOWN,
}