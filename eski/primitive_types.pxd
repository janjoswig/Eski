cimport numpy as np


ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE
ctypedef np.float32_t AVALUE32
ctypedef np.uint8_t ABOOL

cdef struct Constants:
    AVALUE kB
    AVALUE R
    AVALUE u

cpdef Constants make_constants()