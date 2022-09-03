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


cdef inline AVALUE _get_max_abs(AVALUE *ptr, AINDEX n) nogil:
    cdef AVALUE cmax = cfabs(ptr[0])
    cdef AVALUE candidate
    cdef AINDEX i

    for i in range(1, n):
        candidate = cfabs(ptr[i])
        if candidate > cmax:
            cmax = candidate

    return cmax


def get_max_abs(a):
    """Finds the maximum value in an array

    Args:
        a: Array-like

    Returns:
        Maximum
    """
    cdef AVALUE[::1] aview = a
    return _get_max_abs(&aview[0], a.shape[0])


cdef inline void _distance(
        AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr, AINDEX d) nogil:
    """Calculate distance vector

    Args:
       rvptr: Pointer to output distance vector array.
       p1ptr: Pointer to first input position array.
       p2ptr: Pointer to second input position array.
    """

    cdef AINDEX i

    for i in range(d):
        rvptr[i] = p2ptr[i] - p1ptr[i]


cdef inline AVALUE _norm2(AVALUE *rvptr, AINDEX d) nogil:
    cdef AVALUE norm = 0
    cdef AINDEX i

    for i in range(d):
        norm = norm + rvptr[i] * rvptr[i]

    return csqrt(norm)


cdef inline AVALUE _norm2sq(AVALUE *rvptr, AINDEX d) nogil:
    cdef AVALUE norm = 0
    cdef AINDEX i

    for i in range(d):
        norm = norm + rvptr[i] * rvptr[i]

    return norm


cdef inline void _normalise(
        AVALUE *rvptr, AVALUE *rvnptr, AVALUE n, AINDEX d) nogil:
    cdef AINDEX i

    for i in range(d):
        rvnptr[i] = rvptr[i] / n


def euclidean_distance(p1, p2, norm=False):
    """Calculate euclidean distance between two vectors

    Args:
       p1: Coordinates of point 1
       p2: Coordinates of point 2

    Returns:
        Distance
    """

    cdef AVALUE[::1] p1view = p1
    cdef AVALUE[::1] p2view = p2
    cdef AVALUE[::1] rv = np.zeros_like(p1, dtype=P_AVALUE)

    _distance(&rv[0], &p1view[0], &p2view[0], len(p1view))

    if norm:
        return _norm2(&rv[0], len(p1view))
    else:
        return np.asarray(rv)


cdef inline AVALUE _cosangle(
        AVALUE *rvanptr, AVALUE *rvbnptr, AINDEX d) nogil:
    cdef AVALUE costheta = 0
    cdef AINDEX i

    for i in range(d):
        costheta = costheta + rvanptr[i] * rvbnptr[i]

    return costheta


cdef inline AVALUE _derangle(
        AVALUE costheta,
        AVALUE *rvanptr, AVALUE *rvbnptr,
        AVALUE ra, AVALUE rb,
        AVALUE *der1ptr, AVALUE *der2ptr, AVALUE *der3ptr,
        AINDEX d) nogil:
    cdef AVALUE sininv = 0
    cdef AINDEX i

    sininv = 1.0 / csqrt(1.0 - costheta * costheta)
    for i in range(d):
        der1ptr[i] = sininv * (costheta  * rvanptr[i] - rvbnptr[i]) / ra
        der3ptr[i] = sininv * (costheta  * rvbnptr[i] - rvanptr[i]) / rb
        der2ptr[i] = -(der1ptr[i] + der3ptr[i])


cdef inline AVALUE _torsion(
        AVALUE *rvaptr, AVALUE *uptr, AVALUE *vptr, AVALUE rb, AINDEX d) nogil:

    cdef AINDEX i
    cdef AVALUE x = 0
    cdef AVALUE y = 0

    for i in range(d):
        x += rb * vptr[i] * rvaptr[i]
        y += (uptr[i] * vptr[i])

    return catan2(x, y)


cdef inline AVALUE _dertorsion(
    AVALUE *rvaptr, AVALUE *rvbptr, AVALUE *rvcptr,
    AVALUE *uptr, AVALUE *vptr,
    AVALUE rb, AVALUE rbsq, AVALUE usq, AVALUE vsq,
    AVALUE *der1ptr, AVALUE *der2ptr, AVALUE *der3ptr, AVALUE *der4ptr,
    AINDEX d) nogil:

    cdef AINDEX i
    cdef AVALUE abbc = 0
    cdef AVALUE bccd = 0

    for i in range(d):
        abbc = abbc + rvaptr[i] * rvbptr[i]
        bccd = bccd + rvbptr[i] * rvcptr[i]

    for i in range(d):
        der1ptr[i] = (-rb / usq) * uptr[i]
        der4ptr[i] = (rb / vsq) * vptr[i]

        der2ptr[i] = (-abbc / rbsq - 1) * der1ptr[i] + (bccd / rbsq) * der4ptr[i]
        der3ptr[i] = (-bccd / rbsq - 1) * der4ptr[i] + (abbc / rbsq) * der1ptr[i]


cdef inline void _cross3(
        AVALUE *rvaptr, AVALUE *rvbptr, AVALUE *uptr) nogil:
    uptr[0] = rvaptr[1] * rvbptr[2] - rvaptr[2] * rvbptr[1]
    uptr[1] = rvaptr[2] * rvbptr[0] - rvaptr[0] * rvbptr[2]
    uptr[2] = rvaptr[0] * rvbptr[1] - rvaptr[1] * rvbptr[0]