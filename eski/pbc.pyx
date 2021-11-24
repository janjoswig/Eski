import numpy as np

cimport cython
from cython.parallel cimport prange

from eski.primitive_types import P_AINDEX, P_AVALUE


cdef class PBCHandler:

    cdef void _apply_pbc(self, System system) nogil: ...


cdef class NoPBC(PBCHandler):

    cdef void _apply_pbc(self, System system) nogil: pass


cdef class OrthorhombicPBC(PBCHandler):

    def __cinit__(self, bounds):

        self._bounds = bounds

    cdef void _apply_pbc(self, System system) nogil:

        cdef AINDEX index, i, d
        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef AVALUE *configuration = &system._configuration[0]
        cdef AVALUE *bounds = &self._bounds[0]

        for index in prange(system._n_atoms):
            for d in range(dim_per_atom):
                i = index * dim_per_atom + d
                configuration[i] = cython.cmod(configuration[i], bounds[d])


cdef class TriclinicPBC(PBCHandler):

    def __cinit__(self, box):

        self._box = box
        self._boxinv = np.linalg.inv(self._box)

    cdef void _apply_pbc(self, System system) nogil:

        cdef AINDEX index, i, d, d_
        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef AVALUE *configuration = &system._configuration[0]
        cdef AVALUE *box = &self._box[0, 0]
        cdef AVALUE *boxinv = &self._boxinv[0, 0]
        cdef AVALUE rid_f

        for index in prange(system._n_atoms):
            for d in range(dim_per_atom):
                rid_f = 0
                i = index * dim_per_atom + d
                for d_ in range(dim_per_atom):
                    rid_f = rid_f + boxinv[d * dim_per_atom + d_] * configuration[i]
                configuration[i] = rid_f - <np.intp_t>rid_f

                rid_f = 0
                for d_ in range(dim_per_atom):
                    rid_f = rid_f + box[d * dim_per_atom + d_] * configuration[i]
                configuration[i] = rid_f
