cimport numpy as np


ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


cdef AVALUE _euclidean_distance(
    AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr) nogil
