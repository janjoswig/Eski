cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, acos as cacos, cos as ccos, sin as csin

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array
from eski.md cimport System
from eski.metrics cimport _distance, _norm2


cdef class InteractionProvider:
    cdef:
        Interaction _interaction
        AINDEX _n_interactions

    cpdef void _check_consistency(self) except *
    cdef (AINDEX*, AVALUE*) _next_interaction(self) nogil
    cdef void _reset_iteration(self) nogil

cdef class NoProvider(InteractionProvider): pass

cdef class ExplicitProvider(InteractionProvider):
    cdef:
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _n_indices, _n_parameters
        AINDEX _it

cdef class NeighboursProvider(InteractionProvider): pass


cdef class Interaction:

    cdef public:
        AINDEX group
        AINDEX _id
    cdef:
        list _index_names
        list _param_names
        AINDEX _dindex, _dparam
        InteractionProvider _provider

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
