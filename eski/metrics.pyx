import numpy as np

from eski.primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef inline double _random_uniform() nogil:
    cdef double r = rand()
    return r / RAND_MAX


def random_uniform():
    return _random_uniform()


cdef inline double _random_gaussian() nogil:
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * _random_uniform() - 1.0
        x2 = 2.0 * _random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * clog(w)) / w) ** 0.5
    return x1 * w


def random_gaussian():
    return _random_gaussian()


cdef inline AVALUE _get_max(AVALUE *ptr, AINDEX n) nogil:
    cdef AVALUE cmax = ptr[0]
    cdef AINDEX i

    for i in range(1, n):
        if ptr[i] > cmax:
            cmax = ptr[i]

    return cmax


def get_max(a):
    """Finds the maximum value in an array

    Args:
        a: Array-like

    Returns:
        Maximum
    """
    cdef AVALUE[::1] aview = a
    return _get_max(&aview[0], a.shape[0])


cdef inline AVALUE _euclidean_distance(
        AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr, AINDEX d) nogil:
    """Calculate euclidean distance

    Args:
       rvptr: Pointer to output distance vector array.
       p1ptr: Pointer to first input position array.
       p2ptr: Pointer to second input position array.

    Returns:
       Distance
    """

    cdef AINDEX i
    cdef AVALUE r = 0

    for i in range(d):
        rvptr[i] = p1ptr[i] - p2ptr[i]
        r += cpow(rvptr[i], 2)

    return csqrt(r)


def euclidean_distance(p1, p2):
    """Calculate euclidean distance between two vectors

    Args:
       p1: Array-like coordinates of point 1
       p2: Array-like coordinates of point 2

    Returns:
        Distance
    """

    cdef AVALUE[::1] p1view = p1
    cdef AVALUE[::1] p2view = p2
    cdef AVALUE[::1] rv = np.zeros(p1.shape[0], dtype=P_AVALUE)

    return _euclidean_distance(&rv[0], &p1view[0], &p2view[0], p1.shape[0])
