cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog
from libc.math cimport fmax

from eski.primitive_types cimport AINDEX, AVALUE
from eski.atoms cimport InternalAtom, make_internal_atoms
from eski.md cimport System
from eski.metrics cimport _random_gaussian, _get_max
from eski.interactions cimport Interaction


cdef class Driver:
    cdef:
        AVALUE *_parameters
        AINDEX _dparam
        AINDEX _n_parameters

    cdef void _update(self, System system)


cdef class SteepestDescentMinimiser(Driver):
    pass


cdef class EulerIntegrator(Driver):
    pass


cdef class EulerMaruyamaIntegrator(Driver):
    pass