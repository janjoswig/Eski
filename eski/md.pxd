cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog, acos as cacos, cos as ccos, sin as csin, atan2 as catan2

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.primitive_types cimport Constants, make_constants
from eski.interactions cimport Interaction
from eski.drivers cimport Driver
from eski.atoms cimport Atom, InternalAtom, make_internal_atoms
from eski.pbc cimport PBCHandler, NoPBC
from eski.neighbours cimport Neighbours, NoNeighbours
from eski.metrics cimport _random_gaussian, _norm2, _normalise, _cross3, _torsion


cdef class Resources:
    cdef public:
        object _system
    cdef:
        AVALUE *rva
        AVALUE *rvb
        AVALUE *rvc
        AVALUE *rvan
        AVALUE *rvbn
        AVALUE *rvcn
        AVALUE *der1
        AVALUE *der2
        AVALUE *der3
        AVALUE *der4
        AVALUE *u
        AVALUE *v
        AVALUE *com_velocity
        AVALUE[::1] configuration
        AVALUE prev_epot

    cdef AVALUE* allocate_avalue_array(self, AINDEX n)

cdef class System:

    cdef public:
        str desc
        list interactions
        list drivers
        list reporters
        PBCHandler pbc
        Neighbours neighbours

    cdef:
        AVALUE[::1] _configuration
        AVALUE *_configuration_ptr
        AVALUE[::1] _velocities
        AVALUE *_velocities_ptr
        AVALUE[::1] _forces
        AVALUE *_forces_ptr
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
        object __weakref__


    cdef void allocate_atoms(self)
    cdef void reset_forces(self) nogil
    cpdef AVALUE potential_energy(self)
    cdef AVALUE _kinetic_energy(self) nogil
    cdef AVALUE _temperature(self, AVALUE ekin) nogil
    cpdef void add_all_forces(self)
    cdef void _simulate(self, Py_ssize_t n, bint reset_step)
    cpdef AVALUE _get_total_mass(self)
    cdef void _remove_com_velocity(self) nogil
    cdef void _generate_velocities(self, AVALUE T) nogil
    cpdef void reset_resources(self)


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
