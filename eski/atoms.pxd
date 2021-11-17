from eski.primitive_types cimport AINDEX, AVALUE


ctypedef struct internal_atom:
    AINDEX atype_id
    AVALUE mass
    AVALUE charge


ctypedef struct system_support:
    AINDEX n_atoms
    AINDEX n_dim
    AINDEX dim_per_atom


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


cdef void make_internal_atoms(list atoms, internal_atom *_atoms)
