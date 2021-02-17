from numbers import Integral
from typing import Iterable, Mapping
from typing import Union

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

    def __repr__(self):
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
