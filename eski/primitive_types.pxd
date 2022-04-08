cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.pair cimport pair

ctypedef np.intp_t AINDEX
ctypedef np.intp_t* AINDEXPTR
ctypedef np.float64_t AVALUE
ctypedef np.float64_t* AVALUEPTR
ctypedef np.float32_t AVALUE32
ctypedef np.float32_t* AVALUE32PTR
ctypedef np.uint8_t ABOOL
ctypedef np.uint8_t* ABOOLPTR

ctypedef pair[AINDEXPTR, AVALUEPTR] IVPTRPAIR

cdef struct Constants:
    AVALUE kB
    AVALUE R
    AVALUE u

cpdef Constants make_constants()

cdef AVALUE* _allocate_and_fill_avalue_array(AINDEX n, list values)

cdef AINDEX* _allocate_and_fill_aindex_array(AINDEX n, list values)

