from eski.primitive_types cimport AINDEX, AVALUE, ABOOL


ctypedef struct InternalAtom:
    AINDEX atype_id
    AVALUE mass
    AVALUE charge


cdef class Atom:

    cdef public:
        AINDEX aid
        AINDEX resid
        str aname
        str atype
        str element
        str residue
        str chain
        AVALUE mass
        AVALUE charge


cdef void make_internal_atoms(list atoms, InternalAtom *_atoms)
