cimport numpy as np

from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.atoms cimport system_support


ctypedef struct resources:
    AVALUE *rv


cdef class Interaction:

    cdef public:
        AINDEX group
        AINDEX _id
    cdef:
        list _index_names
        list _param_names
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _dindex, _dparam
        AINDEX _n_indices, _n_parameters

    cpdef void add_all_forces(
        self,  AVALUE[::1] configuration,  AVALUE[::1] forces,
        system_support support)

    cdef void _add_all_forces(
        self,  AVALUE *configuration, AVALUE *forces,
        system_support support, resources res) nogil

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            AVALUE *configuration,
            AVALUE *forces,
            system_support support,
            resources res) nogil

    cpdef AVALUE get_total_energy(
        self,  AVALUE[::1] configuration,
        system_support support)

    cdef AVALUE _get_total_energy(
        self,  AVALUE *configuration,
        system_support support, resources res) nogil

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            AVALUE *configuration,
            system_support support,
            resources res) nogil

    cpdef void _check_index_param_consistency(self) except *
    cpdef void _check_interaction_index(self, AINDEX index) except *


cdef resources allocate_resources(system_support support) except *