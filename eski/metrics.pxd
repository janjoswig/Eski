cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE


cdef double _random_uniform() nogil
cdef double _random_gaussian() nogil


cdef AVALUE _euclidean_distance(
    AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr, AINDEX d) nogil
