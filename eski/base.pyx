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
    pass


cdef class Force2p(Force):
    """Base class for two-particle force"""
    cdef AINDEX p1, p2  # atom indices
    cdef AVALUE[::1] rv, fv, ev  # Distance, force, energy vector container

cdef class ForceHarmonicBond(Force2p):
    """Harmonic spring force approximating a chemical bond"""

    cdef AVALUE r0  # Equillibrium bond length (nm)
    cdef AVALUE k   # Force constanct (kJ / (mol nm**2))

    def __cinit__(self, p1, p2, r0, k):
        self.p1 = p1
        self.p2 = p2
        self.r0 = r0
        self.k = k
        self.rv = np.zeros(3)
        self.fv = np.zeros(3)
        self.ev = np.zeros(3)

    def __repr__(self):
        return f"{self.__class__.__name__}(p1={self.p1}, p2={self.p2}, r0={self.r0}, k={self.k})"

    cdef AVALUE force(
            self, AVALUE *structure):
        """Evaluate harmonic bond force

        Args:
            structure: Atom positions.
        
        Returns:
            None; Sets `fv`
        """
        
        cdef AINDEX i
        cdef AVALUE r, f  # Distance, force

        r = euclidean_distance(
            &self.rv[0],
            &structure[self.p1 * 3],
            &structure[self.p2 * 3]
            )
        
        # Force
        f = -self.k * (r - self.r0)
        for i in range(3):
            self.fv[i] = f * self.rv[i] / r

        return f

    @property
    def _fv(self):
        """Expose current force vector"""
        return self.fv

    cpdef f(self, AVALUE[:, ::1] structure):
        """Calculate force and return magnitude and force vector"""
        cdef AVALUE f

        f = self.force(&structure[0, 0])

        return f, np.asarray(self.fv)

    cdef AVALUE energy(
            self, AVALUE *structure):
        """Evaluate harmonic bond energy

        Args:
            structure: Atom positions.
        
        Returns:
            None; Sets `ev`
        """

        cdef AINDEX i
        cdef AVALUE r, e  # Distance, force

        r = euclidean_distance(
            &self.rv[0],
            &structure[self.p1 * 3],
            &structure[self.p2 * 3]
            )

        # Force
        e = self.k / 2 * cpow((r - self.r0), 2)
        for i in range(3):
            self.ev[i] = e * self.rv[i] / r

        return e

    @property
    def _ev(self):
        """Expose current energy vector"""
        return self.ev

    cpdef e(self, AVALUE[:, ::1] structure):
        """Calculate energy and return magnitude and energy vector"""
        cdef AVALUE f

        e = self.energy(&structure[0, 0])

        return e, np.asarray(self.ev)


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
            masses: Atom masses (u).
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
    
    def __cinit__(self, interval):
        self.interval = interval


cdef class ListReporter(Reporter):
    """A reporter that collects data in Python lists"""
    
    cdef list structures
    cdef list velocities
    
    def __cinit__(self, interval):
        self.interval = interval
        self.structures = []
        self.velocities = []
        
    cpdef void report(self, System system):
        self.structures.append(system.structure)
        self.velocities.append(system.velocities)
     
    
cdef class PrintReporter(Reporter):
    """A reporter that prints data"""
    
    def __cinit__(self, interval):
        self.interval = interval
        
    cpdef void report(self, System system):
        print(np.asarray(system.structure))

        
cdef class System:
    """MD system
    
    A system associates a molecular structure (atom positions of one
    or more molecules) and particles velocities with a set of forces
    (bonds, angles etc.) to evaluate.  Particle masses are stored in a
    separate attribute.  Also included is a driver, that
    is a dynamical integrator which propagates positions and velocities. 
    A reporter can be used to communicate data during a simulation.
    """

    cdef AVALUE[:, ::1] structure   # Atom positions
    cdef AVALUE[:, ::1] velocities  # Atom velocities
    cdef AVALUE[::1] masses         # Atom masses
    cdef AINDEX n_atoms             # Number of atoms
    cdef list force_list            # Forces to evaluate
    cdef AVALUE[:, ::1] forces      # Evaluated forces
    cdef IntegratorEuler driver     # MD integrator
    cdef Reporter reporter          # Data reporter
    
    def __cinit__(
            self,
            structure,   # Initial structure
            velocities,  # Initial velocities
            masses,      # Particle masses
            force_list,  # Forces to evaluate
            driver,      # MD integrator
            reporter):   # Data reporter
        self.structure = np.copy(structure)     # Preserve input structure
        self.velocities = np.copy(velocities)   #                velocities
        self.masses = masses
        self.n_atoms = self.structure.shape[0]
        self.force_list = force_list
        self.forces = np.zeros_like(structure)  # Initialise force container
        self.driver = driver
        self.reporter = reporter
        
    def __repr__(self):
        return f"{self.__class__.__name__}(n_atoms={self.n_atoms}, driver={self.driver})"
        
    cdef inline void reset_forces(self) nogil:
        "Reinitialise force vector matrix"
        
        for i in range(self.n_atoms):
            for j in range(3):
                self.forces[i, j] = 0
        
    cdef inline void evaluate_forces(self):
        """Evaluate force vector acting on each atom"""
        
        cdef AINDEX i
        cdef ForceHarmonicBond force
        
        self.reset_forces()
        
        for force in self.force_list:
            force.force(&self.structure[0, 0])
            for i in range(3):
                self.forces[force.p1, i] = self.forces[force.p1, i] + force.fv[i]
                self.forces[force.p2, i] = self.forces[force.p2, i] - force.fv[i]
    
    def step(self, Py_ssize_t n):
        """Perform n MD simulation steps"""

        cdef Py_ssize_t i
        
        for i in range(1, n + 1):
            self.evaluate_forces()
    
            self.driver.update(
                self.n_atoms,
                self.structure,
                self.velocities,
                self.masses,
                self.forces
            )
            if not i % self.reporter.interval:
                self.reporter.report(self)
