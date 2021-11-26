cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.md cimport System


cdef class PBCHandler:

    cdef void _apply_pbc(self, System system) nogil
    cdef void _pbc_distance(
        self, AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr,
        AINDEX dim_per_atom) nogil

cdef class NoPBC(PBCHandler):
    pass


cdef class OrthorhombicPBC(PBCHandler):
    cdef AVALUE[::1] _bounds


cdef class TriclinicPBC(PBCHandler):
    cdef AVALUE[:, ::1] _box
    cdef AVALUE[:, ::1] _boxinv
