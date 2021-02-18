cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE
from eski.atoms cimport internal_atom


cdef class System:

    cdef public:
        str desc
        list forces
        list drivers
    cdef:
        AVALUE[:, ::1] _structure
        AVALUE[:, ::1] _velocities
        AVALUE[:, ::1] _forcevectors
        AINDEX _n_atoms
        internal_atom *_atoms
        AVALUE[:, ::1] _box, _boxinv
        Py_ssize_t _step

    cdef void allocate_atoms(self)
    cdef void reset_forcevectors(self) nogil
    cpdef void step(self, Py_ssize_t n)
