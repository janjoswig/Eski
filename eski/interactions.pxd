cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, acos as cacos, cos as ccos, sin as csin

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.md cimport System
from eski.metrics cimport _distance, _norm2


cdef class Interaction:

    cdef public:
        AINDEX group
        AINDEX _id
        bint requires_gil
    cdef:
        list _index_names
        list _param_names
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _dindex, _dparam
        AINDEX _n_indices, _n_parameters

    cdef AVALUE* _allocate_and_fill_avalue_array(
            self, AINDEX n, list values)

    cdef AINDEX* _allocate_and_fill_aindex_array(
            self, AINDEX n, list values)

    cdef void _add_force(
        self,
        AINDEX *indices,
        AVALUE *parameters,
        System system)

    cdef void _add_force_nogil(
        self,
        AINDEX *indices,
        AVALUE *parameters,
        System system) nogil

    cdef void _add_force_by_index(
        self,
        AINDEX index,
        System system)

    cdef void _add_force_by_index_nogil(
        self,
        AINDEX index,
        System system) nogil

    cdef void _add_all_forces(
        self, System system)

    cdef void _add_all_forces_nogil(
        self,  System system) nogil

    cdef AVALUE _get_energy(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system)

    cdef AVALUE _get_energy_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil

    cdef AVALUE _get_energy_by_index(
        self,
        AINDEX index,
        System system)

    cdef AVALUE _get_energy_by_index_nogil(
        self,
        AINDEX index,
        System system) nogil

    cdef AVALUE _get_total_energy(
        self,  System system)

    cdef AVALUE _get_total_energy_nogil(
        self,  System system) nogil


    cpdef void _check_index_param_consistency(self) except *
    cpdef void _check_interaction_index(self, AINDEX index) except *



cdef class ConstantBias(Interaction):
    pass


cdef class Exclusion(Interaction):
    pass


cdef class HarmonicPositionRestraint(Interaction):
    pass


cdef class HarmonicBond(Interaction):
    pass


cdef class LJ(Interaction):
    pass
