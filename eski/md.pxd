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


cdef class Force:

    cdef public:
        AINDEX group
    cdef:
        AINDEX _id
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _dindex, _dparam
        AINDEX _n_indices, _n_parameters

    cpdef void add_contributions(self, System system)

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors,
            AVALUE *rv,
            AVALUE *fv) nogil


cdef class Driver:
    cdef:
        AVALUE *_parameters

    cdef void update(self, System system)


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


cdef AVALUE _euclidean_distance(
    AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr) nogil
