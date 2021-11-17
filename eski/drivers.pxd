from eski.primitive_types cimport AINDEX, AVALUE
from eski.atoms cimport internal_atom, system_support

cdef class Driver:
    cdef:
        AVALUE *_parameters
        AINDEX _dparam
        AINDEX _n_parameters

    cpdef void update(
            self,
            AVALUE[::1] configuration,
            AVALUE[::1] velocities,
            AVALUE[::1] forces,
            list atoms,
            system_support support)

    cdef void _update(
            self,
            AVALUE *configuration,
            AVALUE *velocities,
            AVALUE *forces,
            internal_atom *atoms,
            system_support support) nogil
