cimport numpy as np
from eski.md cimport System


ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


cdef class Force:

    cdef public:
        AINDEX group
    cdef:
        AINDEX _id
        AINDEX *_indices
        AVALUE *_parameters
        AINDEX _dindex, _dparam
        AINDEX _n_indices, _n_parameters

    cpdef void add_contributions(self, System system)

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors,
            AVALUE *rv,
            AVALUE *fv) nogil
