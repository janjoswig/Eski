cimport numpy as np

from eski.md cimport System

ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


cdef class Driver:
    cdef:
        AVALUE *_parameters
        AINDEX _dparam
        AINDEX _n_parameters

    cdef void update(self, System system)
