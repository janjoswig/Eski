import sys
import warnings

cimport cython
from cython.parallel cimport prange
import numpy as np

from eski.primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef Constants constants = make_constants()


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
            drivers=None,
            reporters=None,
            pbc=None,
            desc=None,
            copy=False):
        """Init docstring"""

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
        self._configuration_ptr =  &self._configuration[0]

        n_dim = self._configuration.shape[0]
        assert n_dim  == n_atoms * dim_per_atom, f"Number of dimensions ({n_dim}) does not match dimensions per atoms ({dim_per_atom})"

        self._n_atoms = n_atoms
        self._n_dim = n_dim
        self._dim_per_atom = dim_per_atom
        self.allocate_atoms()

        if atoms is not None:
            assert len(atoms) == n_atoms, f"Number of atoms in 'configuration' ({n_atoms}) and 'atoms' ({len(atoms)}) does not match"
            make_internal_atoms(atoms, self._atoms)
            self._total_mass = self._get_total_mass()

        else:
            warnings.warn(
                "No `atoms` provided. "
                "Simulation features that require atom data "
                "May not be available.",
                RuntimeWarning
                )
            self._atoms = NULL
            self._total_mass = 0

        if velocities is None:
            velocities = np.zeros_like(self._configuration)

        if velocities.ndim == 2:
            velocities = velocities.reshape(-1)

        elif velocities.ndim != 1:
            raise ValueError(
                f"Parameter `velocities` needs to be a 1D or 2D array "
                f"but has dimensionality {velocities.ndim}"
                )

        assert velocities.shape[0] == configuration.shape[0], f"Shape of 'velocities' {velocities.shape} does not match 'configuration' {configuration.shape[0]}"

        self._velocities = np.array(
            velocities,
            copy=copy,
            dtype=P_AVALUE,
            order="c"
            )
        self._velocities_ptr =  &self._velocities[0]

        self._forces = np.zeros_like(
            configuration,
            dtype=P_AVALUE,
            order="c"
            )
        self._forces_ptr =  &self._forces[0]

        if interactions is None:
            interactions = []
        self.interactions = interactions

        self.set_dof()

        if drivers is None:
            drivers = []
        self.drivers = drivers

        if reporters is None:
            reporters = []
        self.reporters = reporters

        if pbc is None:
            pbc = NoPBC()
        self._pbc = pbc

        self._step = 0
        self._target_step = 0
        self._stop = False

        if desc is None:
            desc = ""
        self.desc = desc

        self._resources = Resources(self)

    def __dealloc__(self):
        if self._atoms != NULL:
            free(self._atoms)

    @property
    def configuration(self):
        return np.asarray(self._configuration)

    def set_configuration(self, value):
        """Set a new configuration

        Maybe important for simulations where the number of particles
        can change
        """

    @property
    def velocities(self):
        return np.asarray(self._velocities)

    @property
    def forces(self):
        return np.asarray(self._forces)

    @property
    def n_atoms(self):
        return self._n_atoms

    @property
    def dim_per_atom(self):
        return self._dim_per_atom

    @property
    def total_mass(self):
        return self._total_mass

    @property
    def step(self):
        return self._step

    @property
    def target_step(self):
        return self._target_step

    @property
    def stop(self):
        return self._stop

    def __repr__(self):
        if self.desc == "":
            desc_str = ""
        else:
            desc_str = f"{self.desc!r}, "

        if self._n_atoms == 1:
            atoms_str = "1 atom"
        else:
            atoms_str = f"{self.n_atoms} atoms"

        dim_str = f" ({self.dim_per_atom}D)"

        return f"{self.__class__.__name__}({desc_str}{atoms_str}{dim_str})"

    @property
    def dof(self):
        return self._dof

    @dof.setter
    def dof(self, v):
        self._dof = v

    def set_dof(
            self, remove_com_trans=True, remove_com_rot=False,
            check_constraints=False):

        self._dof = 3 * self._n_atoms
        if remove_com_trans:
            self.dof -= 3

        if remove_com_rot:
            self.dof -= 3

        # TODO: check for constrained interactions

    def distance(self, AINDEX a, AINDEX b):
        """PBC distance between atoms with indices a and b"""

        cdef AINDEX i
        cdef AVALUE r = 0

        self._pbc._pbc_distance(
            self._resources.rva,
            &self._configuration_ptr[a * self._dim_per_atom],
            &self._configuration_ptr[b * self._dim_per_atom],
            self._dim_per_atom
            )

        for i in range(self._dim_per_atom):
            r += cpow(self._resources.rva[i], 2)

        return csqrt(r)

    cdef void allocate_atoms(self):
        self._atoms = <InternalAtom*>malloc(
            self._n_atoms * sizeof(InternalAtom)
            )

        if self._atoms == NULL:
            raise MemoryError()

    cpdef AVALUE _get_total_mass(self):
        cdef AINDEX i
        cdef AVALUE total_mass = 0

        for i in range(self._n_atoms):
            total_mass += self._atoms[i].mass

        return total_mass

    cdef inline void _remove_com_velocity(self) nogil:
        cdef AINDEX index, d, i
        cdef AVALUE *com_velocity = self._resources.com_velocity

        for index in prange(self._n_atoms):
            for d in range(self._dim_per_atom):
                i = index * self._dim_per_atom + d

                com_velocity[d] = com_velocity[d] + self._velocities[i] * self._atoms[i].mass / self._total_mass

        for index in prange(self._n_atoms):
            for d in range(self._dim_per_atom):
                i = index * self._dim_per_atom + d

                self._velocities[i] = self._velocities[i] - com_velocity[d]

    cdef void _generate_velocities(self, AVALUE T) nogil:
        cdef AVALUE instant_temperature
        cdef AVALUE scale_factor
        cdef AVALUE sigma
        cdef AINDEX index, d, i

        for index in prange(self._n_atoms):

            sigma = csqrt(constants.R * T / self._atoms[index].mass)

            for d in range(self._dim_per_atom):
                i = index * self._dim_per_atom + d

                self._velocities[i] = _random_gaussian() * sigma

        self._remove_com_velocity()

        instant_temperature = self._temperature(-1)
        scale_factor = csqrt(T / instant_temperature)

        for index in prange(self._n_atoms):
            for d in range(self._dim_per_atom):
                i = index * self._dim_per_atom + d

                self._velocities[i] = self._velocities[i] * scale_factor

    def generate_velocities(self, T=300):
        self._generate_velocities(T)

    cdef inline void reset_forces(self) nogil:
        """Reinitialise force vector"""

        cdef AINDEX i

        for i in range(self._n_dim):
            self._forces[i] = 0

    cpdef AVALUE potential_energy(self):
        """Compute the current potential energy of the system"""

        cdef Interaction interaction

        cdef AVALUE energy = 0

        for interaction in self.interactions:
            if not interaction.requires_gil:
                energy += interaction._get_total_energy_nogil(self)
            else:
                energy += interaction._get_total_energy(self)

        return energy

    cdef AVALUE _kinetic_energy(self) nogil:
        """Compute the current kinetic energy of the system"""

        cdef AINDEX index, d, i
        cdef AVALUE energy = 0
        cdef AVALUE vnorm2

        for index in prange(self._n_atoms):
            if self._atoms[index].mass <= 0:
                continue

            vnorm2 = 0
            for d in range(self._dim_per_atom):
                i = index * self._dim_per_atom + d
                vnorm2 = vnorm2 + cpow(self._velocities[i], 2)

            energy += 0.5 * self._atoms[index].mass * vnorm2

        return energy

    def kinetic_energy(self):
        return self._kinetic_energy()

    cdef AVALUE _temperature(self, AVALUE ekin) nogil:
        "Compute current system temperature"

        if ekin < 0:
            ekin = self._kinetic_energy()

        return (2 * ekin) / (self._dof * constants.R)

    def temperature(self, ekin=None):
        if ekin is None: ekin = -1
        return self._temperature(ekin)

    cpdef void add_all_forces(self):

        cdef Interaction interaction

        self.reset_forces()

        for interaction in self.interactions:
            if not interaction.requires_gil:
                interaction._add_all_forces_nogil(self)
            else:
                interaction._add_all_forces(self)

    def simulate(self, n, reset_step=False):
        """Perform a number of MD simulation steps

        Args:
            n: number of steps (set to 0 for no limit)
        Keyword args:
            reset_step: reset cumulative step count to 0
        """
        self._simulate(n, reset_step)

    cdef void _simulate(self, Py_ssize_t n, bint reset_step):
        """Perform a number of MD simulation steps"""

        cdef Driver driver
        cdef Reporter reporter

        if reset_step:
            self._step = 0

        assert n >= 0

        if n == 0:
            self._target_step = sys.maxsize - 1
        else:
            self._target_step = self._step + n
        self._stop = False

        for driver in self.drivers:
            driver._on_startup(self)

        for self._step in range(self._step + 1, self._target_step + 1):

            if self._stop:
                break

            for driver in self.drivers:
                driver._update(self)

            self._pbc._apply_pbc(self)

            for reporter in self.reporters:
                if (self._step % reporter.interval) == 0:
                    reporter.report(self)

        self.reset_resources()

    cpdef void reset_resources(self):
        self._resources = Resources(self)


cdef class Resources:

    def __cinit__(self, System system):

        self.rva = self.allocate_avalue_array(system._dim_per_atom)
        self.rvb = self.allocate_avalue_array(system._dim_per_atom)
        self.rvc = self.allocate_avalue_array(system._dim_per_atom)
        self.der1 = self.allocate_avalue_array(system._dim_per_atom)
        self.der2 = self.allocate_avalue_array(system._dim_per_atom)
        self.der3 = self.allocate_avalue_array(system._dim_per_atom)
        self.com_velocity = self.allocate_avalue_array(system._dim_per_atom)

        self.configuration = np.array([])

        self.prev_epot = 0

    def __dealloc__(self):

        if self.rva != NULL: free(self.rva)
        if self.rvb != NULL: free(self.rvb)
        if self.rvc != NULL: free(self.rvc)
        if self.der1 != NULL: free(self.der1)
        if self.der2 != NULL: free(self.der2)
        if self.der3 != NULL: free(self.der3)
        if self.com_velocity != NULL: free(self.com_velocity)

    cdef AVALUE* allocate_avalue_array(self, AINDEX n):

        cdef AVALUE *ptr
        cdef AINDEX i

        ptr = <AVALUE*>malloc(n * sizeof(AVALUE))

        if ptr == NULL:
            raise MemoryError()

        for i in range(n):
            ptr[i] = 0

        return ptr


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
        cdef str attr

        for attr in self.reported_attrs:
            attr_value = getattr(system, attr)

            if isinstance(attr_value, np.ndarray):
                attr_value = np.copy(attr_value)

            step_output[attr] = attr_value

        self.output.append(step_output)

cdef class PrintReporter(Reporter):

    _default_reported_attrs = ["step", "target_step"]
    _default_format_message = lambda self, args: f"Completed step {args[0]}/{args[1]} ({int(100 * args[0]/args[1])} %)"

    def __init__(
            self, interval, *,
            reported_attrs=None, format_message=None):
        if reported_attrs is None:
            reported_attrs = self._default_reported_attrs
        self.reported_attrs = reported_attrs

        if format_message is None:
            format_message = self._default_format_message
        self.format_message = format_message

    cpdef void report(self, System system):
        print(
            self.format_message(
                [getattr(system, attr) for attr in self.reported_attrs]
                ),
            end="\r"
            )