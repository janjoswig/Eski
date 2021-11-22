cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.md cimport System
from eski.metrics cimport _euclidean_distance


cdef class Interaction:

    cdef public:
        AINDEX group
        AINDEX _id
    cdef:
        list _index_names
        list _param_names
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _dindex, _dparam
        AINDEX _n_indices, _n_parameters

    cdef void _add_all_forces(
        self,  System system) nogil

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            System system) nogil

    cdef AVALUE _get_total_energy(
        self,  System system) nogil

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            System system) nogil

    cpdef void add_all_forces(
            self, System system)

    cpdef void add_force_by_index(
            self,
            AINDEX index,
            System system)

    cpdef AVALUE get_total_energy(
        self,  System system)

    cpdef AVALUE get_energy_by_index(
            self,
            AINDEX index,
            System system)

    cpdef void _check_index_param_consistency(self) except *
    cpdef void _check_interaction_index(self, AINDEX index) except *
