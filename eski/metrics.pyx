import numpy as np
cimport numpy as np

from libc.math cimport sqrt as csqrt, pow as cpow
from eski.primitive_types import P_AINDEX, P_AVALUE


cdef inline AVALUE _euclidean_distance(
        AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr) nogil:
    """Calculate euclidean distance in 3D

    Args:
       rvptr: Pointer to output distance vector array.
       p1ptr: Pointer to first input position array.
       p2ptr: Pointer to second input position array.

    Returns:
       Distance
    """

    cdef AINDEX i
    cdef AVALUE r = 0

    for i in range(3):
        rvptr[i] = p1ptr[i] - p2ptr[i]
        r += cpow(rvptr[i], 2)

    return csqrt(r)


def euclidean_distance(p1, p2):
    """Calculate euclidean distance in 3D

    Args:
       p1: Array-like coordinates of point 1
       p2: Array-like coordinates of point 2

    Returns:
        Distance
    """

    cdef AVALUE[::1] p1view = p1
    cdef AVALUE[::1] p2view = p2
    cdef AVALUE[::1] rv = np.zeros(3, dtype=P_AVALUE)

    return _euclidean_distance(&rv[0], &p1view[0], &p2view[0])
