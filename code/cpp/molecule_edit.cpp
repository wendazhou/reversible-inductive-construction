#include <algorithm>
#include <iterator>

#include <RDBoost/python.h>
#include <RDBoost/Wrap.h>

#include <GraphMol/RDKitBase.h>
#include <GraphMol/RDKitQueries.h>
#include <GraphMol/MonomerInfo.h>

#include <GraphMol/AtomIterators.h>
#include <GraphMol/BondIterators.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>

#include <boost/python.hpp>

#include <vector>
#include <set>


namespace py = boost::python;

namespace {

void copy_edit_mol(RDKit::ROMol const &mol, RDKit::RWMol& result) {
    for (auto it = mol.beginAtoms(), e = mol.endAtoms(); it != e; ++it) {
        auto atom = new RDKit::Atom((*it)->getAtomicNum());
        atom->setFormalCharge((*it)->getFormalCharge());

        result.addAtom(atom, true, true);
    }

    for (auto it = mol.beginBonds(), e = mol.endBonds(); it != e; ++it) {
        auto a1_idx = (*it)->getBeginAtomIdx();
        auto a2_idx = (*it)->getEndAtomIdx();
        result.addBond(a1_idx, a2_idx, (*it)->getBondType());
    }

    result.updatePropertyCache();
}

} // namespace

BOOST_PYTHON_MODULE(molecule_edit) {
    py::def("copy_edit_mol_impl", &copy_edit_mol, py::arg("mol"), py::arg("target"));
}

