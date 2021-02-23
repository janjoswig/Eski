cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE


cdef class Force:

    cdef public:
        AINDEX group
    cdef:
        AINDEX _id
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _dindex, _dparam
        AINDEX _n_indices, _n_parameters
        AVALUE[::1] rv
        AVALUE[::1] fv

    cpdef void add_contributions(
        self,  AVALUE[:, ::1] structure,  AVALUE[:, ::1] forcevectors)

    cdef void _add_contributions(
        self,  AVALUE *structure,  AVALUE *forcevectors) nogil

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors) nogil

    cpdef void _check_index_param_consistency(self) except *
    cpdef void _check_interaction_index(self, AINDEX index) except *