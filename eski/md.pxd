cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.interactions cimport Interaction, resources
from eski.drivers cimport Driver
from eski.atoms cimport Atom, internal_atom, make_internal_atoms, system_support

cdef class System:

    cdef public:
        str desc
        list interactions
        list drivers

    cdef:
        AVALUE[::1] _configuration
        AVALUE[::1] _velocities
        AVALUE[::1] _forces
        system_support _support
        internal_atom *_atoms
        # AVALUE[:, ::1] _box, _boxinv
        AVALUE[::1] _bounds
        bint _use_pbc
        Py_ssize_t _step

    cdef void allocate_atoms(self)
    cdef void reset_forces(self) nogil
    cpdef void step(self, Py_ssize_t n)
