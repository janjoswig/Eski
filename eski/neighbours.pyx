from cython.parallel cimport prange

from eski.primitive_types import P_AINDEX, P_AVALUE


cdef class Neighbours:

    _param_names = []
    _param_defaults = {}

    def __cinit__(self, parameters: list):

        cdef AINDEX i
        cdef AVALUE param

        self._n_parameters = len(parameters)

        self._parameters = <AVALUE*>malloc(
            self._n_parameters * sizeof(AVALUE)
            )
        if self._parameters == NULL:
            raise MemoryError()

        for i, param in enumerate(parameters):
            self._parameters[i] = param

    def __init__(self, *args, **kwargs):
        self._check_param_consistency()

    def _check_param_consistency(self):
        cdef AINDEX dparam = len(self._param_names)

        if self._n_parameters != dparam:
            numerus_expect = "parameter" if dparam == 1 else "parameters"
            numerus_given = "was" if self._n_parameters == 1 else "were"

            raise ValueError(
                f"driver {type(self).__name__!r} "
                f"takes {dparam} {numerus_expect} "
                f"but {self._n_parameters} {numerus_given} given"
                )

    cdef void _update(self, System system) nogil: ...

    def update(self, System system):
        self._update(system)

    cdef bint needs_update(self, System system) nogil: ...


cdef class NeighboursVerletBruteForce(Neighbours):

    _param_names = ["cutoff", "buffer"]
    _param_defaults = {}

    @property
    def neighbourlist(self):
        return self._neighbourlist

    @property
    def n_neighbours_positions(self):
        return self._n_neighbours_pos

    cdef void _update(self, System system) nogil:
        cdef AINDEX a, b
        cdef AVALUE r
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1
        cdef AVALUE *c2
        cdef AVALUE cutoff = self._parameters[0]
        cdef AVALUE buffer = self._parameters[1]
        cdef AVALUE *rv = system._resources.rva
        cdef AINDEX *n_neighbours_pos
        cdef AINDEX n_neighbours

        self._neighbourlist.clear()
        self._neighbourlist.push_back(0)
        self._n_neighbours_pos.push_back(0)

        for a in range(system._n_atoms):
            for b in prange(a + 1, system._n_atoms):
                c1 = &system._configuration_ptr[a * d]
                c2 = &system._configuration_ptr[b * d]

                system._pbc._pbc_distance(rv, c1, c2, d)
                r = _norm2(rv, d)

                if r <= buffer:
                    n_neighbours += 1
                    self._neighbourlist.push_back(b)
                    self._neighbourlist[self._n_neighbours_pos[a]] += 1

            self._neighbourlist.push_back(0)
            self._n_neighbours_pos.push_back(
                self._n_neighbours_pos[a] + self._neighbourlist[self._n_neighbours_pos[a]] + 1
                )

