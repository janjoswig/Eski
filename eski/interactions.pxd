cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport (
    sqrt as csqrt, pow as cpow,
    acos as cacos, cos as ccos, sin as csin
)
from libcpp.vector cimport vector

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array
from eski.md cimport System
from eski.neighbours import Neighbours
from eski.metrics cimport _distance, _norm2, _norm2sq, _normalise, _cosangle, _derangle, _cross3, _torsion, _dertorsion


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


cdef class NeighboursProvider(InteractionProvider):
    cdef:
        System _system
        Table _table

cdef class Table:
    cdef AVALUE* _get_parameters(self, AINDEX* indices) nogil


cdef class DummyTable(Table):
    cdef:
        AVALUE parameters[2]


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

cdef class ConstantBias(Interaction): pass
cdef class Exclusion(Interaction): pass
cdef class Stabilizer(Interaction): pass
cdef class HarmonicPositionRestraint(Interaction): pass
cdef class HarmonicRepulsion(Interaction): pass
cdef class HarmonicBond(Interaction): pass
cdef class HarmonicAngle(Interaction): pass
cdef class CosineHarmonicAngle(Interaction): pass
cdef class HarmonicTorsion(Interaction): pass
cdef class LJ(Interaction): pass
