cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.primitive_types cimport Constants, make_constants
from eski.interactions cimport Interaction
from eski.drivers cimport Driver
from eski.atoms cimport Atom, InternalAtom, make_internal_atoms
from eski.pbc cimport PBCHandler, NoPBC
from eski.metrics cimport _random_gaussian


cdef class Resources:
    cdef:
        AVALUE *rv
        AVALUE *rvb
        AVALUE *rvc
        AVALUE *der1
        AVALUE *der2
        AVALUE *der3
        AVALUE[::1] configuration_b

    cdef AVALUE* allocate_avalue_array(self, AINDEX n)

cdef class System:

    cdef public:
        str desc
        list interactions
        list custom_interactions
        list drivers
        list reporters
        PBCHandler _pbc

    cdef:
        AVALUE[::1] _configuration
        AVALUE[::1] _velocities
        AVALUE[::1] _forces
        AINDEX _n_atoms
        AINDEX _n_dim
        AINDEX  _dim_per_atom
        AVALUE _total_mass
        AINDEX _dof
        InternalAtom *_atoms
        Resources _resources
        Py_ssize_t _step
        Py_ssize_t _target_step
        bint _stop

    cdef void allocate_atoms(self)
    cdef void reset_forces(self) nogil
    cpdef AVALUE potential_energy(self)
    cdef AVALUE _kinetic_energy(self) nogil
    cdef AVALUE _temperature(self, AVALUE ekin) nogil
    cpdef void add_all_forces(self)
    cpdef void simulate(self, Py_ssize_t n)
    cpdef AVALUE _get_total_mass(self)
    cdef void _remove_com_velocity(self) nogil
    cdef void _generate_velocities(self, AVALUE T) nogil


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
        object format_message
