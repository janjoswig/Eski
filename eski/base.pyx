import numpy as np
cimport numpy as np

from libc.math cimport sqrt as csqrt, pow as cpow


ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


cdef inline AVALUE euclidean_distance(
        AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr) nogil:
    """Calculate euclidean distance

    Args:
       rvptr: Pointer to output distance vector array.
       p1ptr: Pointer to first input position array.
       p2ptr: Pointer to second input position array.

    Returns:
       Distance
    """

    cdef AINDEX i
    cdef AVALUE r = 0

    for i in range(3):
        rvptr[i] = p1ptr[i] - p2ptr[i]
        r += cpow(rvptr[i], 2)

    return csqrt(r)


cdef class Force:
    """Base class for force to evaluate"""

    cdef AINDEX _id
    cdef AINDEX group

    def __cinit__(self):
        self._id = 99
        self.group = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(name={FORCE_ID_NAME_MAPPING[self._id]}, group={self.group})"

    @property
    def id(self):
       return self._id

    @property
    def group(self):
       return self.group

    @group.setter
    def group(self, group):
       self.group = group


cdef class ForceHarmonicBond(Force):
    """Harmonic spring force approximating a chemical bond"""

    def __cinit__(self):
        self._id = 1
        self.group = 0

    cdef AVALUE force(
            self,
            AINDEX p1, AINDEX p2,
            AVALUE r0, AVALUE k,
            AVALUE *structure, AVALUE *rv, AVALUE *fv) nogil:
        """Evaluate harmonic bond force

        Args:
            p1: Index of atom 1.
            p2: Index of atom 2.
            r0: Equillibrium bond length (nm)
            k: Force constanct (kJ / (mol nm**2))
            structure: Pointer to atom positions.
            rv: Pointer to array used for the distance vector.
            fv: Pointer to array used for the force vector.

        Returns:
            Force (kJ / (mol nm))
        """

        cdef AINDEX i
        cdef AVALUE r, f  # Distance, force

        r = euclidean_distance(
            rv,
            &structure[p1 * 3],
            &structure[p2 * 3]
            )

        # Force
        f = -k * (r - r0)
        for i in range(3):
            fv[i] = f * rv[i] / r

        return f

    cpdef f(self, AINDEX p1, AINDEX p2, AVALUE r0, AVALUE k, AVALUE[:, ::1] structure):
        """Calculate force and return magnitude and force vector"""

        cdef AVALUE f
        cdef AVALUE[::1] rv = np.zeros(3)
        cdef AVALUE[::1] fv = np.zeros(3)

        f = self.force(p1, p2, r0, k, &structure[0, 0], &rv[0], &fv[0])

        return f, np.asarray(fv)

    cdef AVALUE energy(
            self, AINDEX p1, AINDEX p2, AVALUE r0, AVALUE k, AVALUE *structure, AVALUE *rv, AVALUE *ev):
        """Evaluate harmonic bond energy

        Args:
            p1: Index of atom 1.
            p2: Index of atom 2.
            r0: Equillibrium bond length (nm)
            k: Force constanct (kJ / (mol nm**2))
            structure: Pointer to atom positions.
            rv: Pointer to array used for the distance vector.
            ev: Pointer to array used for the energy vector.

        Returns:
            Energy (kJ / mol)
        """

        cdef AINDEX i
        cdef AVALUE r, e  # Distance, force

        r = euclidean_distance(
            rv,
            &structure[p1 * 3],
            &structure[p2 * 3]
            )

        # Force
        e = k / 2 * cpow((r - r0), 2)
        for i in range(3):
            ev[i] = e * rv[i] / r

        return e

    cpdef e(self, AINDEX p1, AINDEX p2, AVALUE r0, AVALUE k, AVALUE[:, ::1] structure):
        """Calculate energy and return magnitude and energy vector"""

        cdef AVALUE f
        cdef AVALUE[::1] rv = np.zeros(3)
        cdef AVALUE[::1] ev = np.zeros(3)

        e = self.energy(p1, p2, r0, k, &structure[0, 0], &rv[0], &ev[0])

        return e, np.asarray(ev)


cdef class ForceLJ(Force):
    """Lennard-Jones interaction"""

    def __cinit__(self):
        self._id = 2
        self.group = 0

    cdef AVALUE force(
            self,
            AINDEX p1, AINDEX p2,
            AINDEX at1, AINDEX at2,
            AVALUE s1, AVALUE s2,
            AVALUE e1, AVALUE e2,
            AVALUE *structure, AVALUE *rv, AVALUE *fv) nogil:
        """Evaluate harmonic bond force

        Args:
            p1: Index of atom 1.
            p2: Index of atom 2.
            s1: Sigma for p1.
            s2: Sigma for p2.
            e1: Epsilon for p1.
            e2: Epsilon for p2.
            structure: Pointer to atom positions.
            rv: Pointer to array used for the distance vector.
            fv: Pointer to array used for the force vector.

        Returns:
            Force (kJ / (mol nm))
        """

        cdef AINDEX i
        cdef AVALUE r, f, s, e  # Distance, force

        r = euclidean_distance(
            rv,
            &structure[p1 * 3],
            &structure[p2 * 3]
            )

        # Lorentz-Berthelot combination
        if at1 == at2:
            s = s1
            e = e1
        else:
            s = (s1 + s2) / 2
            e = csqrt(e1 * e2)

        # Force
        f = 48 * e * (cpow(s, 12) / cpow(r, 13) - cpow(s, 6) / cpow(r, 7)) # 4ϵ[σ12r12−σ6r6]
        for i in range(3):
            fv[i] = f * rv[i] / r

        return f


FORCE_ID_NAME_MAPPING = {
    1: "HarmonicBond",
    2: "LJ",
    99: "Force",
}

FORCE_NAME_ID_MAPPING = {v: k for k, v in FORCE_ID_NAME_MAPPING.items()}

FORCE_ID_CLASS_MAPPING = {
    force().id: force for force in (ForceHarmonicBond, ForceLJ,)
}

ctypedef fused FORCE:
    ForceHarmonicBond
    ForceLJ


cdef class Integrator:
    pass


cdef class IntegratorEuler:
    cdef AVALUE dt

    def __cinit__(self, dt):
        self.dt = dt

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self.dt})"

    cdef void update(
            self,
            AINDEX n_atoms,
            AVALUE[:, ::1] structure,
            AVALUE[:, ::1] velocities,
            AVALUE[::1] masses,
            AVALUE[:, ::1] forces):
        """Perform Euler step

        Args:
            structure: Atom positions (nm).
            velocites: Atom velocities (nm/ps).
            forces: List of forces to evaluate (kJ / (mol nm)).
        """

        cdef AINDEX index
        cdef AINDEX d

        for index in range(n_atoms):
            for d in range(3):
                structure[index, d] = structure[index, d] + velocities[index, d] * self.dt + forces[index, d] * 1.661e-12 * self.dt**2 / (2 * masses[index])
                velocities[index, d] = velocities[index, d] + forces[index, d] * self.dt / masses[index]


cdef class Reporter:
    """Base class for reporters"""

    cdef Py_ssize_t interval

    def __cinit__(self, interval, *args, **kwargs):
        self.interval = interval

    @property
    def interval(self):
        return self.interval

cdef class ListReporter(Reporter):
    """A reporter that collects data in Python lists"""

    cdef list structures
    cdef list velocities

    def __cinit__(self, interval):
        self.interval = interval
        self.structures = []
        self.velocities = []

    cpdef void report(self, System system):
        self.structures.append(np.copy(np.asarray(system.structure)))
        self.velocities.append(np.copy(np.asarray(system.velocities)))

    @property
    def structures(self):
        return self.structures

    @property
    def velocities(self):
        return self.velocities


cdef class PrintReporter(Reporter):
    """A reporter that prints data"""

    def __cinit__(self, interval):
        self.interval = interval

    cpdef void report(self, System system):
        print(np.asarray(system.structure))


cdef class LogReporter(Reporter):
    """A reporter that prints information to a log file"""

    cdef str file
    cdef bint initialised

    def __cinit__(self, interval, file):
        self.interval = interval
        self.file = file
        self.initialised = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self.interval}, file={self.file})"

    cpdef void report(self, System system):
        if self.initialised:
            with open(self.file, "a") as logfile:
                logfile.write(f"{system._step}\n")
        else:
            with open(self.file, "w") as logfile:
                logfile.write("Step    Performance\n")
                logfile.write(f"{system._step}\n")
            self.initialised = True

    @property
    def file(self):
        return self.file


cdef class XYZReporter(Reporter):
    """A reporter that writes structures to a xyz-file"""

    cdef str file

    def __cinit__(self, interval, file):
        self.interval = interval
        self.file = file

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self.interval}, file={self.file})"

    cpdef void report(self, System system):
        with open(self.file, "a") as xyzfile:
           xyzfile.write(f"{system.n_atoms}\n")
           xyzfile.write(f"Atomic structure at step = {system._step}, automatically written by Eski\n")
           for index, position in enumerate(system.structure):
               xyzfile.write(f'{system.topology["element"][index]}    ')
               xyzfile.write(f"{position[0]:2.10f}    {position[1]:2.10f}    {position[2]:2.10f}\n")

    @property
    def file(self):
        return self.file


cdef class System:
    """MD system

    A system associates a molecular structure (atom positions of one
    or more molecules), particle velocities and a molecular topology
    with a set of forces
    (bonds, angles etc.) to evaluate.  Also included is a driver, that
    is a dynamical integrator which propagates positions and velocities.
    A reporter can be used to communicate data during a simulation.
    """

    cdef AVALUE[:, ::1] _structure    # Atom positions
    cdef AVALUE[:, ::1] _velocities   # Atom velocities
    cdef dict topology                # Topology
    cdef AINDEX[::1] atomtypes        # Atom types (IDs)
    cdef dict ATOM_ID_TYPE_MAPPING
    cdef dict ATOM_TYPE_ID_MAPPING
    cdef AVALUE[:, ::1] box
    cdef AVALUE[:, ::1] boxinv

    cdef AINDEX n_atoms             # Number of atoms
    cdef AVALUE[::1] masses         # Atom types
    cdef AVALUE[::1] charges        # Atom types
    cdef dict force_map             # Forces to evaluate
    cdef list force_list            # List of Force instances
    cdef dict forcefield
    cdef AVALUE[:, ::1] forces      # Evaluated forces
    cdef IntegratorEuler driver     # MD integrator
    cdef list reporters             # Data reporters

    cdef Py_ssize_t _step
    cdef AVALUE[::1] rv
    cdef AVALUE[::1] fv

    def __cinit__(
            self,
            structure,   # Initial structure
            velocities,  # Initial velocities
            topology,    # Particle types, names, elements, residues
            box,         # Box vectors as columns of a matrix
            masses,      # 
            charges,     #
            force_map,   # Forces to evaluate
            forcefield,  # Dicttionary defining forcefield parameters
            driver,      # MD integrator
            reporters):   # Data reporter
        self._structure = np.copy(structure)     # Preserve input structure
        self._velocities = np.copy(velocities)   #                velocities
        self.n_atoms = structure.shape[0]
        self.masses = masses
        self.charges = charges
        self.make_topology(topology)
        self.make_forces(force_map, forcefield)
        self.forces = np.zeros_like(structure)  # Initialise force container
        self.box = np.copy(box)
        self.boxinv = np.linalg.inv(box)
        self.driver = driver
        self.reporters = reporters

        self._step = 0
        self.rv  = np.zeros(3)  # Reserved space for distance vector
        self.fv  = np.zeros(3)  #                    force vector

    def __repr__(self):
        return f"{self.__class__.__name__}(n_atoms={self.n_atoms}, driver={self.driver})"

    @property
    def structure(self):
        return np.asarray(self._structure)

    @property
    def velocities(self):
        return np.asarray(self._velocities)

    def make_topology(self, dict topology):
        self.topology = topology

        atomtype_id = 0
        self.ATOM_TYPE_ID_MAPPING = {}
        for atom in topology["type"]:
            if atom in self.ATOM_TYPE_ID_MAPPING:
                continue
            self.ATOM_TYPE_ID_MAPPING[atom] = atomtype_id
            atomtype_id += 1

        self.ATOM_ID_TYPE_MAPPING = {
            v: k
            for k, v in self.ATOM_TYPE_ID_MAPPING.items()
            }

        self.atomtypes = np.asarray([
            self.ATOM_TYPE_ID_MAPPING[x]
            for x in topology["type"]
            ])

    def make_forces(self, force_map, forcefield):
        self.force_map = {
            FORCE_NAME_ID_MAPPING[k]: v
            for k, v in force_map.items()
            }

        self.force_list = [
            FORCE_ID_CLASS_MAPPING[force_id]()
            for force_id in self.force_map.keys()
            ]

        self.forcefield = {}
        for force, params in forcefield.items():
            force = FORCE_NAME_ID_MAPPING[force]
            self.forcefield[force] = {}
            for k, v in params.items():
                if isinstance(k, str):
                    self.forcefield[force][self.ATOM_TYPE_ID_MAPPING[k]] = v
                elif isinstance(k, tuple):
                    # TODO: Check also for int in tuples
                    k = tuple(
                        self.ATOM_TYPE_ID_MAPPING[p]
                        if isinstance(p, str) else p
                        for p in k
                        )
                    self.forcefield[force][k] = v
                elif isinstance(k, int):
                    self.forcefield[force][k] = v
                else:
                    raise TypeError()

    cdef inline void reset_forces(self) nogil:
        "Reinitialise force vector matrix"

        for i in range(self.n_atoms):
            for j in range(3):
                self.forces[i, j] = 0

    def evaluate_force(self, FORCE force):
        self._evaluate_force(force)

    cdef void _evaluate_force(self, FORCE force):
        cdef AINDEX i
        cdef AINDEX p1, p2, at1, at2
        cdef AVALUE r0, k, s1, s2, e1, e2

        if FORCE is ForceHarmonicBond:
            for p1, p2  in self.force_map[force._id]:
                at1 = self.atomtypes[p1]
                at2 = self.atomtypes[p2]
                r0 = self.forcefield[force._id][(at1, at2)][0]
                k = self.forcefield[force._id][(at1, at2)][1]

            force.force(
                p1, p2, r0, k,
                &self._structure[0, 0],
                &self.rv[0], &self.fv[0]
                )
            for i in range(3):
                self.forces[p1, i] = self.forces[p1, i] + self.fv[i]
                self.forces[p2, i] = self.forces[p2, i] - self.fv[i]

        elif FORCE is ForceLJ:
            for p1, p2 in self.force_map[force._id]:
                at1 = self.atomtypes[p1]
                at2 = self.atomtypes[p2]
                s1 = self.forcefield[force._id][at1][0]
                s2 = self.forcefield[force._id][at2][0]
                e1 = self.forcefield[force._id][at1][1]
                e2 = self.forcefield[force._id][at2][1]

                # print("Force parameters:\n", p1, p2, at1, at2, s1, s2, e1, e2)

            force.force(
                p1, p2, at1, at2, s1, s2, e1, e2,
                &self._structure[0, 0],
                &self.rv[0], &self.fv[0]
                )

            for i in range(3):
                self.forces[p1, i] = self.forces[p1, i] + self.fv[i]
                self.forces[p2, i] = self.forces[p2, i] - self.fv[i]

    def apply_pbc(self):
        # Convert positions to fractial coordinates
        fractial = np.dot(self.boxinv, self._structure.T)

        # Put particles back in the box
        fractial_pbc = fractial - np.floor(fractial)

        # Convert back to real positions
        self._structure = np.asarray(np.dot(self.box, fractial_pbc).T, order="c", dtype=np.float64)

    def step(self, Py_ssize_t n):
        """Perform n MD simulation steps"""

        self._step = 0

        for self._step in range(1, n + 1):
            # Evaluate forces
            for force in self.force_list:
                self.evaluate_force(force)

            # Propagate
            self.driver.update(
                self.n_atoms,
                self._structure,
                self._velocities,
                self.masses,
                self.forces
            )

            # Periodic boundary conditions
            self.apply_pbc()

            # Report
            for reporter in self.reporters:
                if not self._step % reporter.interval:
                    reporter.report(self)
