cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, acos as cacos, cos as ccos, sin as csin

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array
from eski.md cimport System
from eski.metrics cimport _distance, _norm2


cdef class InteractionProvider:
    cdef:
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _n_indices, _n_parameters

    cdef (AINDEX*, AVALUE*) get_interaction_by_index(
        self, AINDEX index, Interaction interaction) nogil

    cpdef void _check_index_param_consistency(
        self, Interaction interaction) except *
    cpdef void _check_interaction_index(
        self, AINDEX index, Interaction interaction) except *
    cdef AINDEX _n_interactions(
        self, Interaction interaction) nogil

cdef class NoProvider(InteractionProvider): pass

cdef class ExplicitProvider(InteractionProvider): pass

cdef class NeighboursProvider(InteractionProvider): pass


cdef class Interaction:

    cdef public:
        AINDEX group
        AINDEX _id
        InteractionProvider provider
    cdef:
        list _index_names
        list _param_names
        AINDEX _dindex, _dparam

    cdef void _add_force(
        self,
        AINDEX *indices,
        AVALUE *parameters,
        System system) nogil

    cdef void _add_all_forces(
        self, System system) nogil

    cdef AVALUE _get_energy(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil

    cdef AVALUE _get_total_energy(
        self, System system) nogil

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
