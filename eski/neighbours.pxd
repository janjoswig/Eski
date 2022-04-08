cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.metrics cimport _norm2
from eski.md cimport System


cdef class Neighbours:
    cdef:
        AVALUE *_parameters
        AINDEX _n_parameters

    cdef void _update(self, System system) nogil
    cdef bint needs_update(self, System system) nogil

cdef class NeighboursVerletBruteForce(Neighbours):
    cdef:
        vector[AINDEX] _neighbourlist
        vector[AINDEX] _n_neighbours_pos
