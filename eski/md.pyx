from typing import Iterable, Mapping
from typing import Union

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from eski.primitive_types import P_AINDEX, P_AVALUE
from eski.forces cimport Force
from eski.drivers cimport Driver
from eski.atoms cimport Atom, internal_atom, make_internal_atoms


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
            assert len(atoms) == self._n_atoms
            make_internal_atoms(atoms, self._atoms)

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

    cdef inline void reset_forcevectors(self) nogil:
        """Reinitialise force vector matrix"""

        cdef AINDEX i, j

        for i in range(self._n_atoms):
            for j in range(3):
                self._forcevectors[i, j] = 0

    cpdef void step(self, Py_ssize_t n):
        """Perform a number of MD simulation steps"""

        cdef Force force
        cdef Driver driver

        self._step = 0

        for self._step in range(1, n + 1):

            self.reset_forcevectors()

            for force in self.forces:
                force._add_contributions(
                    &self._structure[0, 0],
                    &self._forcevectors[0, 0],
                    )

            for driver in self.drivers:
                driver._update(
                    &self._structure[0, 0],
                    &self._velocities[0, 0],
                    &self._forcevectors[0, 0],
                    self._atoms,
                    self._n_atoms
                    )

            # self.apply_pbc

            # for reporter in self.reporters:
            #     if self._step % reporter.interval == 0:
            #         reporter.report(self)
