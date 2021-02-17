cimport numpy as np


ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE

ctypedef struct internal_atom:
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


cdef class System:

    cdef public:
        str desc
    cdef:
        AVALUE[:, ::1] _structure
        AVALUE[:, ::1] _velocities
        AVALUE[:, ::1] _forcevectors
        AINDEX _n_atoms
        internal_atom *_atoms
        AVALUE[:, ::1] _box, _boxinv
        dict atype_id_mapping
        list forces
        list drivers
        Py_ssize_t _step
        AVALUE[::1] rv
        AVALUE[::1] fv

    cdef void allocate_atoms(self)
    cdef void reset_forcevectors(self) nogil
