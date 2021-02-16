from numbers import Integral
from typing import Iterable

import numpy as np
cimport numpy as np

from libc.math cimport sqrt as csqrt, pow as cpow
from libc.stdlib cimport malloc, free


P_AVALUE = np.float64
P_AINDEX = np.intp


cdef class Atom:
    """Bundels topologic information"""

    def __cinit__(
            self,
            aname=None,
            atype=None,
            aid=None,
            element=None,
            residue="UNK",
            resid=None,
            mass=0,
            charge=0):

        if aname is None:
            aname = ""
        self.aname = aname

        if atype is None:
            atype = aname
        self.atype = atype

        if element is None:
            element = aname
        self.element = element

        self.residue = residue

        self.mass = mass
        self.charge = charge

    def __repr__(self):
        attributes = (
            f"(aname={self.aname}, "
            f"atype={self.atype}, "
            f"element={self.element}, "
            f"residue={self.residue}, "
            f"mass={self.mass}, "
            f"charge={self.charge})"
            )
        return f"{self.__class__.__name__}{attributes}"


cdef class Force:
    """Base class for force to evaluate

    If Cython supported abstract classes and/or virtual methods, this
    would be an abstract base class for the force interface.  This
    class is not meant to be initialised.

    Args:
        indices: Iterable of particle indices for which this force
            should be evaluated.
        parameters: Iterable of force parameters.
    """

    def __cinit__(
            self,
            indices: Iterable[int],
            parameters: Iterable[float],
            *args,
            **kwargs):

        self._n_indices = len(indices)
        self._n_parameters = len(parameters)

        cdef AINDEX i, index
        cdef AVALUE param

        self._indices = <AINDEX*>malloc(
            self._n_indices * sizeof(AINDEX)
            )
        if self._indices == NULL:
            raise MemoryError()

        self._parameters = <AVALUE*>malloc(
            self._n_parameters * sizeof(AVALUE)
            )
        if self._parameters == NULL:
            raise MemoryError()

        for i, index in enumerate(indices):
            self._indices[i] = index

        for i, param in enumerate(parameters):
            self._parameters[i] = param

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 0
        self._dindex = 1
        self._dparam = 0

        self._check_index_param_consistency()

    def __repr__(self):
        return f"{self.__class__.__name__}(group={self.group})"

    @property
    def id(self):
       return self._id

    @property
    def n_interactions(self):
        return self._n_indices / self._dindex

    def __dealloc__(self):
        if self._indices != NULL:
            free(self._indices)

        if self._parameters != NULL:
            free(self._parameters)

    def _check_index_param_consistency(self):
        """Raise error if indices and parameters do not match"""

        if (self._n_indices % self._dindex) > 0:
            raise ValueError(
                f"Wrong number of 'indices'; must be multiple of {self._dindex}"
                )

        if self._dparam == 0:
            if self._n_parameters == 0:
                return
            raise ValueError(
                "Force takes no parameters"
                )

        if (self._n_parameters % self._dparam) > 0:
            raise ValueError(
                f"Wrong number of 'parameters'; must be multiple of {self._dparam}"
                )

        len_no_match = (
            (self._n_indices / self._dindex) !=
            (self._n_parameters / self._dparam)
        )
        if len_no_match:
            raise ValueError(
                "Length of 'indices' and 'parameters' does not match"
                )

    def get_interaction(self, AINDEX index):
        raise NotImplementedError

    cpdef void add_contributions(self, System system):
        raise NotImplementedError

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors,
            AVALUE *rv,
            AVALUE *fv) nogil:
        raise NotImplementedError


cdef class ForceHarmonicBond(Force):
    """Harmonic spring force approximating a chemical bond"""

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 1
        self._dindex = 2
        self._dparam = 2

        self._check_index_param_consistency()

    def get_interaction(self, AINDEX index):
        """Return info for interaction

        Returns:
            Dictionary with keys
                p1: Index of atom 1,
                p2: Index of atom 2,
                r0: Equillibrium bond length (nm),
                k: Force constanct (kJ / (mol nm**2))
        """

        if (index < 0) or (index >= self.n_interactions):
            raise ValueError(
                "Interaction index out of range"
                )

        info = {
            "p1": self._indices[index * self._dindex],
            "p2": self._indices[index * self._dindex + 1],
            "r0": self._parameters[index * self._dparam],
            "k": self._parameters[index * self._dparam + 1]
            }
        return info

    cpdef void add_contributions(self, System system):
        cdef AINDEX index

        for index in range(self._n_indices / self._dindex):
            self._add_contribution(
                index,
                &system._structure[0, 0],
                &system._forcevectors[0, 0],
                &system.rv[0],
                &system.fv[0]
                )

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors,
            AVALUE *rv,
            AVALUE *fv) nogil:
        """Evaluate harmonic bond force

        Args:
            index: Index of interaction
            structure: Pointer to atom positon array

        Returns:
            Force (kJ / (mol nm))
        """

        cdef AINDEX i
        cdef AVALUE r, f
        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AINDEX p2 = self._indices[index * self._dindex + 1]
        cdef AVALUE r0 = self._parameters[index * self._dparam]
        cdef AVALUE k = self._parameters[index * self._dparam + 1]
        cdef AVALUE *fv1
        cdef AVALUE *fv2

        r = _euclidean_distance(
            rv,
            &structure[p1 * 3],
            &structure[p2 * 3]
            )

        fv1 = &forcevectors[p1 * 3]
        fv2 = &forcevectors[p2 * 3]

        f = -k * (r - r0)
        for i in range(3):
            fv[i] = f * rv[i] / r
            fv1[i] += fv[i]
            fv2[i] -= fv[i]

        return


cdef class Driver:

    def __cinit__(self, **kwargs):
        pass

    cdef void update(self, System system):
        pass


cdef class EulerIntegrator(Driver):

    def __cinit__(self, **kwargs):

        self._parameters = <AVALUE*>malloc(
            1 * sizeof(AVALUE)
            )
        if self._parameters == NULL:
            raise MemoryError()

        self._parameters[0] = kwargs.get("dt", 0.001)

    def __dealloc__(self):

        if self._parameters != NULL:
            free(self._parameters)

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self._parameters[0]})"

    cdef void update(self, System system):

        cdef AINDEX index, d

        for index in range(system._n_atoms):
            for d in range(3):
                system._structure[index, d] = (
                    system._structure[index, d]
                    + system._velocities[index, d] * self._parameters[0]
                    + system._forcevectors[index, d] * 1.661e-12
                    * self._parameters[0]**2 / (2 * system._atoms[index].mass)
                    )
                system._velocities[index, d] = (
                    system._velocities[index, d]
                    + system._forcevectors[index, d]
                    * self._parameters[0] / system._atoms[index].mass
                    )


cdef class System:
    """MD System"""

    def __cinit__(
            self,
            structure,
            velocities=None,
            atoms=None,
            forces=None,
            drivers=None,
            box=None,
            desc=None):

        if desc is None:
            desc = ""
        self.desc = desc

        self._structure = np.array(
            structure,
            copy=True,
            dtype=P_AVALUE,
            order="c"
            )

        self._n_atoms = self._structure.shape[0]
        self.allocate_atoms()

        if velocities is None:
            velocities = np.zeros_like(structure)

        self._velocities = np.array(
            velocities,
            copy=True,
            dtype=P_AVALUE,
            order="c"
            )

        self._forcevectors = np.zeros_like(
            structure,
            dtype=P_AVALUE,
            order="c"
            )

        if atoms is not None:
            self.make_atoms(atoms)

        if forces is None:
            forces = []

        if forces is None:
            forces = []
        self.forces = forces

        if drivers is None:
            drivers = []
        self.drivers = drivers

        if box is None:
            # TODO: Check for invalid box
            self._box = np.zeros((3, 3), dtype=P_AVALUE)
            self._boxinv = np.array(self._box, copy=True, dtype=P_AVALUE)

        else:
            self._box = np.array(
                box, copy=True, dtype=P_AVALUE
                )
            self._boxinv = np.linalg.inv(self._box)

        self._step = 0
        self.rv = np.zeros(3, dtype=P_AVALUE)
        self.fv = np.zeros(3, dtype=P_AVALUE)

    def __dealloc__(self):
        if self._atoms != NULL:
            free(self._atoms)

    @property
    def structure(self):
        return np.asarray(self._structure)

    @property
    def velocities(self):
        return np.asarray(self._velocities)

    @property
    def forcevectors(self):
        return np.asarray(self._forcevectors)

    @property
    def n_atoms(self):
        return self._n_atoms

    @property
    def box(self):
        return np.asarray(self._box)

    def __str__(self):
        if self.desc == "":
            desc_str = ""
        else:
            desc_str = f"{self.desc!r}, "

        if self._n_atoms == 1:
            atoms_str = "1 atom"
        else:
            atoms_str = f"{self._n_atoms} atoms"

        return f"{self.__class__.__name__}({desc_str}{atoms_str})"

    cdef void allocate_atoms(self):
        self._atoms = <internal_atom*>malloc(
            self._n_atoms * sizeof(internal_atom)
            )

        if self._atoms == NULL:
            raise MemoryError()

    def make_atoms(self, atoms: Iterable) -> None:
        """Creates internal array of atoms

        Maps the atom type of atoms to an internal id, keeps mass and
        charge, and discards the rest.

        Args:
            atoms: An iterable of :obj:`Atom` instances
                or equivalent types that provide `atype`, `mass`, and
                `charge` attributes.
        """

        assert self.n_atoms == len(atoms)

        cdef AINDEX index, atype_id = 0
        self.atype_id_mapping = {}

        for index, atom in enumerate(atoms):
            if atom.atype not in self.atype_id_mapping:
                self.atype_id_mapping[atom.atype] = atype_id
                atype_id += 1

            self._atoms[index] = internal_atom(
                atype_id=self.atype_id_mapping[atom.atype],
                mass=atom.mass,
                charge=atom.charge
                )

    cdef inline void reset_forcevectors(self) nogil:
        """Reinitialise force vector matrix"""

        cdef AINDEX i, j

        for i in range(self._n_atoms):
            for j in range(3):
                self._forcevectors[i, j] = 0

    def step(self, Py_ssize_t n):
        """Perform a number of MD simulation steps"""

        self._step = 0

        for self._step in range(1, n + 1):

            self.reset_forcevectors()

            for force in self.forces:
                force.add_contributions(self)

            for driver in self.drivers:
                driver.update(self)

            # self.apply_pbc

            # for reporter in self.reporters:
            #     if self._step % reporter.interval == 0:
            #         reporter.report(self)

cdef inline AVALUE _euclidean_distance(
        AVALUE *rvptr, AVALUE *p1ptr, AVALUE *p2ptr) nogil:
    """Calculate euclidean distance in 3D

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


def euclidean_distance(p1, p2):
    """Calculate euclidean distance in 3D

    Args:
       p1: Array-like coordinates of point 1
       p2: Array-like coordinates of point 2

    Returns:
        Distance
    """

    cdef AVALUE[::1] p1view = p1
    cdef AVALUE[::1] p2view = p2
    cdef AVALUE[::1] rv = np.zeros(3, dtype=P_AVALUE)

    return _euclidean_distance(&rv[0], &p1view[0], &p2view[0])
