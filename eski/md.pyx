import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from eski.primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class System:
    """Representing a simulated (molecular) system"""

    def __cinit__(
            self,
            configuration,
            *,
            dim_per_atom,
            velocities=None,
            atoms=None,
            interactions=None,
            drivers=None,
            reporters=None,
            bounds=None,
            desc=None):

        if desc is None:
            desc = ""
        self.desc = desc

        self._configuration = np.array(
            configuration,
            copy=True,
            dtype=P_AVALUE,
            order="c"
            )

        n_dim = self._configuration.shape[0]
        n_atoms = n_dim // dim_per_atom
        assert n_dim  == n_atoms * dim_per_atom, "Number of dimensions does not match dimensions per atoms"

        self._support = system_support(n_atoms, n_dim, dim_per_atom)
        self.allocate_atoms()

        if atoms is not None:
            assert len(atoms) == n_atoms
            make_internal_atoms(atoms, self._atoms)

        if velocities is None:
            velocities = np.zeros_like(configuration)

        self._velocities = np.array(
            velocities,
            copy=True,
            dtype=P_AVALUE,
            order="c"
            )

        self._forces = np.zeros_like(
            configuration,
            dtype=P_AVALUE,
            order="c"
            )

        if interactions is None:
            interactions = []
        self.interactions = interactions

        if drivers is None:
            drivers = []
        self.drivers = drivers

        if reporters is None:
            reporters = []
        self.reporters = reporters

        if bounds is None:
            self._bounds = np.zeros(dim_per_atom)
            self._use_pbc = False
        else:
            self._bounds = bounds
            self._use_pbc = True

        self._step = 0

    def __dealloc__(self):
        if self._atoms != NULL:
            free(self._atoms)

    @property
    def configuration(self):
        return np.asarray(self._configuration)

    @property
    def velocities(self):
        return np.asarray(self._velocities)

    @property
    def forces(self):
        return np.asarray(self._forces)

    @property
    def n_atoms(self):
        return self._support.n_atoms

    @property
    def dim_per_atom(self):
        return self._support.dim_per_atom

    @property
    def bounds(self):
        return np.asarray(self._bounds)

    def __repr__(self):
        if self.desc == "":
            desc_str = ""
        else:
            desc_str = f"{self.desc!r}, "

        if self._support.n_atoms == 1:
            atoms_str = "1 atom"
        else:
            atoms_str = f"{self.n_atoms} atoms"

        dim_str = f" ({self.dim_per_atom}D)"

        return f"{self.__class__.__name__}({desc_str}{atoms_str}{dim_str})"

    cdef void allocate_atoms(self):
        self._atoms = <internal_atom*>malloc(
            self._support.n_atoms * sizeof(internal_atom)
            )

        if self._atoms == NULL:
            raise MemoryError()

    cdef inline void reset_forces(self) nogil:
        """Reinitialise force vector"""

        cdef AINDEX i

        for i in range(self._support.n_dim):
            self._forces[i] = 0

    cpdef void step(self, Py_ssize_t n):
        """Perform a number of MD simulation steps"""

        cdef Interaction interaction
        cdef Driver driver
        cdef Reporter reporter

        cdef resources res = allocate_resources(self._support)

        self._step = 0

        for self._step in range(1, n + 1):

            self.reset_forces()

            for interaction in self.interactions:
                interaction._add_all_forces(
                    &self._configuration[0],
                    &self._forces[0],
                    self._support,
                    res
                    )

            for driver in self.drivers:
                driver._update(
                    &self._configuration[0],
                    &self._velocities[0],
                    &self._forces[0],
                    &self._atoms[0],
                    self._support
                    )

            # self.apply_pbc

            # for reporter in self.reporters:
            #     if self._step % reporter.interval == 0:
            #         reporter.report(self)

            for reporter in self.reporters

        # TODO: Deallocation function
        if res.rv != NULL:
            free(res.rv)