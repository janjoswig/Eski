cdef class Atom:
    """Bundels topologic information"""

    def __cinit__(
            self,
            aname=None,
            atype=None,
            aid=None,
            element=None,
            residue="UNK",
            resid=None,
            mass=0,
            charge=0):

        if aname is None:
            aname = ""
        self.aname = aname

        if atype is None:
            atype = aname
        self.atype = atype

        if element is None:
            element = aname
        self.element = element

        self.residue = residue

        self.mass = mass
        self.charge = charge

    def __repr__(self):
        attributes = (
            f"(aname={self.aname}, "
            f"atype={self.atype}, "
            f"element={self.element}, "
            f"residue={self.residue}, "
            f"mass={self.mass}, "
            f"charge={self.charge})"
            )
        return f"{self.__class__.__name__}{attributes}"


cdef void make_internal_atoms(list atoms, InternalAtom *_atoms):

    cdef AINDEX index, atype_id = 0
    cdef dict atype_id_mapping = {}
    cdef Atom atom

    for index, atom in enumerate(atoms):
        if atom.atype not in atype_id_mapping:
            atype_id_mapping[atom.atype] = atype_id
            atype_id += 1

        _atoms[index] = InternalAtom(
            atype_id=atype_id_mapping[atom.atype],
            mass=atom.mass,
            charge=atom.charge
            )
