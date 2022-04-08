cimport numpy as np
from libc.stdlib cimport malloc, free

ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE
ctypedef np.float32_t AVALUE32
ctypedef np.uint8_t ABOOL


cdef struct Constants:
    AVALUE kB
    AVALUE R
    AVALUE u

cpdef Constants make_constants()

cdef AVALUE* _allocate_and_fill_avalue_array(AINDEX n, list values)

cdef AINDEX* _allocate_and_fill_aindex_array(AINDEX n, list values)

