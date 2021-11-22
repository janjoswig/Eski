cimport numpy as np

from libc.stdlib cimport malloc, free

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.interactions cimport Interaction
from eski.drivers cimport Driver
from eski.atoms cimport Atom, InternalAtom, make_internal_atoms


ctypedef struct Resources:
    AVALUE *rv


cdef class System:

    cdef public:
        str desc
        list interactions
        list custom_interactions
        list drivers
        list reporters

    cdef:
        AVALUE[::1] _configuration
        AVALUE[::1] _velocities
        AVALUE[::1] _forces
        AINDEX _n_atoms
        AINDEX _n_dim
        AINDEX  _dim_per_atom
        InternalAtom *_atoms
        Resources _resources
        AVALUE[::1] _bounds
        bint _use_pbc
        Py_ssize_t _step
        Py_ssize_t _target_step

    cdef void allocate_atoms(self)
    cdef void allocate_resources(self)
    cdef void reset_forces(self) nogil
    cpdef AVALUE potential_energy(self)
    cpdef void add_all_forces(self)
    cpdef void simulate(self, Py_ssize_t n)


cdef class Reporter:

    cdef public:
        Py_ssize_t interval

    cpdef void reset(self)
    cpdef void report(self, System system)


cdef class ListReporter(Reporter):

    cdef public:
        list output
        list reported_attrs


cdef class PrintReporter(Reporter):

    cdef public:
        list reported_attrs
        str message_template
