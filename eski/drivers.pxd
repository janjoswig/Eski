cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog
from libc.math cimport fmax

from eski.primitive_types cimport AINDEX, AVALUE
from eski.primitive_types cimport Constants, make_constants
from eski.atoms cimport InternalAtom, make_internal_atoms
from eski.md cimport System
from eski.metrics cimport _random_gaussian, _get_max_abs
from eski.interactions cimport Interaction


cdef class Driver:
    cdef:
        AVALUE *_parameters
        AINDEX _n_parameters

    cdef void _update(self, System system)
    cdef void _on_startup(self, System system)


cdef class SteepestDescentMinimiser(Driver):
    cdef:
        AVALUE _adjusted_tau

cdef class EulerIntegrator(Driver): pass
cdef class VerletIntegrator(Driver): pass
cdef class EulerMaruyamaIntegrator(Driver): pass
cdef class EulerCromerIntegrator(Driver): pass