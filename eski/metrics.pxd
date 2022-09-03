cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog, atan2 as catan2, fabs as cfabs

from eski.primitive_types cimport AINDEX, AVALUE


cdef double _random_uniform() nogil
cdef double _random_gaussian() nogil

cdef AVALUE _get_max(AVALUE *ptr, AINDEX n) nogil
cdef AVALUE _get_max_abs(AVALUE *ptr, AINDEX n) nogil

cdef void _distance(
    AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr, AINDEX d) nogil

cdef AVALUE _norm2(AVALUE *rvptr, AINDEX d) nogil
cdef AVALUE _norm2sq(AVALUE *rvptr, AINDEX d) nogil

cdef void _normalise(AVALUE *rvptr, AVALUE *rvnptr, AVALUE n, AINDEX d) nogil

cdef AVALUE _cosangle(AVALUE *rvanptr, AVALUE *rvbnptr, AINDEX d) nogil
cdef AVALUE _derangle(
    AVALUE costheta,
    AVALUE *rvanptr, AVALUE *rvbnptr,
    AVALUE ra, AVALUE rb,
    AVALUE *der1ptr, AVALUE *der2ptr, AVALUE *der3ptr,
    AINDEX d) nogil

cdef AVALUE _torsion(
    AVALUE *rvaptr, AVALUE *uptr, AVALUE *vptr, AVALUE rb, AINDEX d) nogil
cdef AVALUE _dertorsion(
    AVALUE *rvaptr, AVALUE *rvbptr, AVALUE *rvcptr,
    AVALUE *uptr, AVALUE *vptr,
    AVALUE rb, AVALUE rbsq, AVALUE usq, AVALUE vsq,
    AVALUE *der1ptr, AVALUE *der2ptr, AVALUE *der3ptr, AVALUE *der4ptr,
    AINDEX d) nogil

cdef void _cross3(AVALUE *rvaptr, AVALUE *rvbptr, AVALUE *uptr) nogil