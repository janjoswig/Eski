import numpy as np

cimport cython
from cython.parallel cimport prange

from libc.math cimport floor, fabs, round

from eski.primitive_types import P_AINDEX, P_AVALUE


cdef class PBCHandler:

    cdef void _apply_pbc(self, System system) nogil: ...
    cdef void _pbc_distance(
            self, AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr,
            AINDEX dim_per_atom) nogil: ...

cdef class NoPBC(PBCHandler):

    cdef void _apply_pbc(self, System system) nogil: pass
    cdef void _pbc_distance(
            self, AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr,
            AINDEX dim_per_atom) nogil:

        cdef AINDEX i

        for i in range(dim_per_atom):
            rvptr[i] = p1ptr[i] - p2ptr[i]


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
                configuration[i] = configuration[i] - floor(configuration[i])

    cdef void _pbc_distance(
            self, AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr,
            AINDEX dim_per_atom) nogil:

        cdef AINDEX i

        for i in range(dim_per_atom):
            rvptr[i] = p1ptr[i] - p2ptr[i]
            rvptr[i] -= self._bounds[i] * round(rvptr[i] / self._bounds[i])


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
                configuration[i] = rid_f - floor(rid_f)

                rid_f = 0
                for d_ in range(dim_per_atom):
                    rid_f = rid_f + box[d * dim_per_atom + d_] * configuration[i]
                configuration[i] = rid_f

    cdef void _pbc_distance(
            self, AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr,
            AINDEX dim_per_atom) nogil:

        cdef AINDEX i, d
        cdef AVALUE *box = &self._box[0, 0]
        cdef AVALUE *boxinv = &self._boxinv[0, 0]
        cdef AVALUE rid_f, rjd_f, rijd_f

        for i in prange(dim_per_atom):
            rvptr[i] = 0

            rid_f = 0
            rjd_f = 0
            for d in range(dim_per_atom):
                rid_f = rid_f + boxinv[i * dim_per_atom + d] * p1ptr[i]
                rjd_f = rjd_f + boxinv[i * dim_per_atom + d] * p2ptr[i]
            rijd_f = rid_f - rjd_f
            rijd_f = rijd_f - round(rijd_f)

            for d in range(dim_per_atom):
                rvptr[i] = rvptr[i] + box[i * dim_per_atom + d] * rijd_f
