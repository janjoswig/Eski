cimport cython
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
            dim_per_atom=None,
            velocities=None,
            atoms=None,
            interactions=None,
            custom_interactions=None,
            drivers=None,
            reporters=None,
            bounds=None,
            desc=None,
            copy=False):

        if configuration.ndim == 2:
            n_atoms, dim_per_atom = configuration.shape
            configuration = configuration.reshape(-1)

        elif configuration.ndim == 1:
            if dim_per_atom is None:
                raise ValueError(
                    "Parameter `dim_per_atom` is required if "
                    "configuration.ndim == 1"
                    )

            if dim_per_atom > 0:
                n_atoms = configuration.shape[0] // dim_per_atom
            else:
                n_atoms = 0
        else:
            raise ValueError(
                f"Parameter `configuration` needs to be a 1D or 2D array "
                f"but has dimensionality {configuration.ndim}"
                )

        self._configuration = np.array(
            configuration,
            copy=copy,
            dtype=P_AVALUE,
            order="c"
            )

        n_dim = self._configuration.shape[0]
        assert n_dim  == n_atoms * dim_per_atom, "Number of dimensions does not match dimensions per atoms"

        self._support = system_support(n_atoms, n_dim, dim_per_atom)
        self.allocate_atoms()

        if atoms is not None:
            assert len(atoms) == n_atoms
            make_internal_atoms(atoms, self._atoms)

        if velocities is None:
            velocities = np.zeros_like(self._configuration)

        if velocities.ndim == 2:
            velocities = velocities.reshape(-1)

        elif velocities.ndim != 1:
            raise ValueError(
                f"Parameter `velocities` needs to be a 1D or 2D array "
                f"but has dimensionality {velocities.ndim}"
                )

        assert velocities.shape[0] == configuration.shape[0]

        self._velocities = np.array(
            velocities,
            copy=copy,
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

        if custom_interactions is None:
            custom_interactions = []
        self.custom_interactions = custom_interactions

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
        self._target_step = 0

        if desc is None:
            desc = ""
        self.desc = desc

    def __dealloc__(self):
        if self._atoms != NULL:
            free(self._atoms)

    @property
    def configuration(self):
        return np.array(self._configuration, copy=True)

    @property
    def velocities(self):
        return np.array(self._velocities, copy=True)

    @property
    def forces(self):
        return np.array(self._forces, copy=True)

    @property
    def n_atoms(self):
        return self._support.n_atoms

    @property
    def dim_per_atom(self):
        return self._support.dim_per_atom

    @property
    def bounds(self):
        return np.array(self._bounds, copy=True)

    @property
    def step(self):
        return self._step

    @property
    def target_step(self):
        return self._target_step

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

    cpdef void simulate(self, Py_ssize_t n):
        """Perform a number of MD simulation steps"""

        cdef Interaction interaction
        cdef object custom_interaction
        cdef Driver driver
        cdef Reporter reporter

        cdef resources res = allocate_resources(self._support)

        self._step = 0
        self._target_step += n

        for self._step in range(1, n + 1):

            self.reset_forces()

            for interaction in self.interactions:
                interaction._add_all_forces(
                    &self._configuration[0],
                    &self._forces[0],
                    self._support,
                    res
                    )

            for custom_interaction in self.custom_interactions:
                custom_interaction.add_all_forces(self)

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

            for reporter in self.reporters:
                if cython.cmod(self._step, reporter.interval) == 0:
                    reporter.report(self)

        # TODO: Deallocation function
        if res.rv != NULL:
            free(res.rv)


cdef class Reporter:
    """Base class for simulation reporters"""

    def __cinit__(self, interval, *, **kwargs):
        self.interval = interval

    cpdef void reset(self): ...
    cpdef void report(self, System system): ...


cdef class ListReporter(Reporter):

    _default_reported_attrs = ["configuration"]

    def __init__(self, interval, *, reported_attrs=None):
        if reported_attrs is None:
            reported_attrs = self._default_reported_attrs
        self.reported_attrs = reported_attrs

        self.reset()

    cpdef void reset(self):
        self.output = []

    cpdef void report(self, System system):
        cdef dict step_output = {}

        for attr in self.reported_attrs:
            step_output[attr] = getattr(system, attr)

        self.output.append(step_output)

cdef class PrintReporter(Reporter):

    _default_reported_attrs = ["step", "target_step"]
    _default_message_template = "Completed step {}/{}"

    def __init__(
            self, interval, *,
            reported_attrs=None, message_template=None):
        if reported_attrs is None:
            reported_attrs = self._default_reported_attrs
        self.reported_attrs = reported_attrs

        if message_template is None:
            message_template = self._default_message_template
        self.message_template = message_template

    cpdef void report(self, System system):

        print(
            self.message_template.format(
                *[getattr(system, attr) for attr in self.reported_attrs]
                ), end="\r"
            )
