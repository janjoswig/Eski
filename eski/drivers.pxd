from eski.primitive_types cimport AINDEX, AVALUE
from eski.atoms cimport internal_atom

cdef class Driver:
    cdef:
        AVALUE *_parameters
        AINDEX _dparam
        AINDEX _n_parameters

    cpdef void update(
            self,
            AVALUE[:, ::1] structure,
            AVALUE[:, ::1] velocities,
            AVALUE[:, ::1] forcevectors,
            list atoms,
            AINDEX n_atoms)

    cdef void _update(
            self,
            AVALUE *structure,
            AVALUE *velocities,
            AVALUE *forcevectors,
            internal_atom *atoms,
            AINDEX n_atoms) nogil
