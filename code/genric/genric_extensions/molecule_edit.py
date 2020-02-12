from . import _molecule_edit as impl

from rdkit import Chem

def copy_edit_mol(mol):
    new_mol = Chem.RWMol()
    impl.copy_edit_mol_impl(mol, new_mol)
    return new_mol
