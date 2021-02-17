cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE


cdef AVALUE _euclidean_distance(
    AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr) nogil
