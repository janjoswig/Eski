cimport numpy as np

from eski.md cimport System
from eski.primitive_types cimport AINDEX, AVALUE


cdef class Driver:
    cdef:
        AVALUE *_parameters
        AINDEX _dparam
        AINDEX _n_parameters

    cdef void update(self, System system)
