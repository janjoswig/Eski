from eski.primitive_types cimport AINDEX, AVALUE, ABOOL
from eski.md cimport System
from eski.pbc cimport PBCHandler


cdef class Explorer:

    cdef public:
        Grid _grid


cdef class Grid:

    cdef:
        AINDEX *_indices
        AVALUE *_resolutions

    cdef void configuration_to_cellid(self, System system)
