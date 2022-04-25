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
        if type(self) is Neighbours:
            raise RuntimeError(
                f"Cannot instantiate abstract class {type(self)}"
                )
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

    cdef bint needs_update(self, System system) nogil:
        return False

    def get_n_neighbours(self, AINDEX index):
        """Return number of neighbours for a given atom index
        Note:
            No bounds check is performed so make sure that `index`
            lies between 0 and n_atoms - 1.
        """

        return self._neighbourlist[self._n_neighbours_pos[index]]

    cdef AINDEX* _next_interaction(self) nogil: ...
    cdef void _reset_iteration(self) nogil: ...
    cdef AINDEX _get_n_interactions(self) nogil: ...


cdef class NoNeighbours(Neighbours):
    pass


cdef class NeighboursVerletBruteForceLinear(Neighbours):

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

                system.pbc._pbc_distance(rv, c1, c2, d)
                r = _norm2(rv, d)

                if r <= buffer:
                    n_neighbours += 1
                    self._neighbourlist.push_back(b)
                    self._neighbourlist[self._n_neighbours_pos[a]] += 1

            self._neighbourlist.push_back(0)
            self._n_neighbours_pos.push_back(
                self._n_neighbours_pos[a] + self._neighbourlist[self._n_neighbours_pos[a]] + 1
                )

    cdef AINDEX* _next_interaction(self) nogil:
        """warning: function returns address of local variable"""
        #cdef AINDEX indices[2]
        #
        #while self._subit == self._neighbourlist[self._it]:
        #    self._it = self._it + 1
        #    self._subit = 0
        #
        #self._subit = self._subit + 1
        #indices[0] = self._it
        #indices[1] = self._neighbourlist[self._it + self._subit]
        #
        #return &indices[0]

    cdef void _reset_iteration(self) nogil:
        self._it = 0
        self._subit = 0

    cdef AINDEX _get_n_interactions(self) nogil:
        return self._neighbourlist.size() - self._n_neighbours_pos.size()


cdef class NeighboursVerletBruteForceSparse(Neighbours):

    _param_names = ["cutoff", "buffer"]
    _param_defaults = {}

    @property
    def neighbourlist(self):
        return self._neighbourlist

    cdef void _update(self, System system) nogil:
        cdef AINDEX a, b
        cdef AVALUE r
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1
        cdef AVALUE *c2
        cdef AVALUE cutoff = self._parameters[0]
        cdef AVALUE buffer = self._parameters[1]
        cdef AVALUE *rv = system._resources.rva

        self._neighbourlist.clear()

        for a in range(system._n_atoms):
            for b in prange(a + 1, system._n_atoms):
                c1 = &system._configuration_ptr[a * d]
                c2 = &system._configuration_ptr[b * d]

                system.pbc._pbc_distance(rv, c1, c2, d)
                r = _norm2(rv, d)

                if r <= buffer:
                    self._neighbourlist.push_back(a)
                    self._neighbourlist.push_back(b)


    cdef AINDEX* _next_interaction(self) nogil:
        self._it = self._it + 1
        return &self._neighbourlist[self._it * 2]

    cdef void _reset_iteration(self) nogil:
        self._it = -1

    cdef AINDEX _get_n_interactions(self) nogil:
        return self._neighbourlist.size() // 2