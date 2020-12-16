import numpy as np
cimport numpy as np


ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


cdef class System:
    """"""

    cdef AVALUE[:, ::1] _structure
    cdef AVALUE[:, ::1] _velocities
    cdef public str desc

    def __cinit__(
            self,
            structure,
            velocities=None,
            desc=None):

        if desc is None:
            desc = ""
        self.desc = desc

        self._structure = np.array(
            structure,
            copy=True,
            dtype=np.float64,
            order="c"
            )

        if velocities is None:
            velocities = np.zeros_like(structure)

        self._velocities = np.array(
            velocities,
            copy=True,
            dtype=np.float64,
            order="c"
            )

    @property
    def structure(self):
        return np.asarray(self._structure)

    @property
    def velocities(self):
        return np.asarray(self._velocities)