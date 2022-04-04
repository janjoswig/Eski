cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog

from eski.primitive_types cimport AINDEX, AVALUE


cdef double _random_uniform() nogil
cdef double _random_gaussian() nogil

cdef AVALUE _get_max(AVALUE *ptr, AINDEX n) nogil

cdef AVALUE _euclidean_distance(
    AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr, AINDEX d) nogil
