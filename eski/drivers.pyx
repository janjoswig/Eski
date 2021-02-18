from typing import Iterable, Mapping
from typing import Union

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from eski.primitive_types import P_AINDEX, P_AVALUE
from eski.atoms cimport internal_atom, make_internal_atoms


cdef class Driver:
    """Base class for drivers to propagate

    If Cython supported abstract classes and/or virtual methods, this
    would be an abstract base class for the driver interface.  This
    class is not meant to be initialised.

    Args:
        parameters: Iterable of driver parameters
    """

    _param_names = []

    def __cinit__(self, parameters: Iterable[float]):

        cdef AINDEX i
        cdef AVALUE param

        self._n_parameters = len(parameters)

        self._parameters = <AVALUE*>malloc(
            self._n_parameters * sizeof(AVALUE)
            )
        if self._parameters == NULL:
            raise MemoryError()

        for i, param in enumerate(parameters):
            self._parameters[i] = param

    def __dealloc__(self):

        if self._parameters != NULL:
            free(self._parameters)

    def __init__(self, *args, **kwargs):
        self._dparam = 0
        self._check_param_consistency()

    def __repr__(self):
        param_repr = ", ".join(
            [
                f"{name}={value}"
                for name, value in self.parameters.items()
            ]
        )
        return f"{self.__class__.__name__}({param_repr})"

    @classmethod
    def from_mapping(cls, parameters: Mapping[str, float]):
        parameter_list = []
        for name in cls._param_names:
            parameter_list.append(parameters[name])

        return cls(parameter_list)

    @property
    def parameters(self):
        pgenerator = (
            self._parameters[index]
            for index in range(self._n_parameters)
            )
        return dict(zip(self._param_names, pgenerator))

    cpdef void update(
            self,
            AVALUE[:, ::1] structure,
            AVALUE[:, ::1] velocities,
            AVALUE[:, ::1] forcevectors,
            list atoms,
            AINDEX n_atoms):
        NotImplemented

    cdef void _update(
            self,
            AVALUE *structure,
            AVALUE *velocities,
            AVALUE *forcevectors,
            internal_atom *atoms,
            AINDEX n_atoms) nogil:
        NotImplemented

    def _check_param_consistency(self):
        if self._n_parameters != self._dparam:
            numerus_expect = "parameter" if self._dparam == 1 else "parameters"
            numerus_given = "was" if self._n_parameters == 1 else "were"

            raise ValueError(
                f"driver {type(self).__name__!r} "
                f"takes {self._dparam} {numerus_expect} "
                f"but {self._n_parameters} {numerus_given} given"
                )


cdef class EulerIntegrator(Driver):

    _param_names = ["dt"]

    def __init__(self, *args, **kwargs):
        self._dparam = 1
        self._check_param_consistency()

    cpdef void update(
        self,
        AVALUE[:, ::1] structure,
        AVALUE[:, ::1] velocities,
        AVALUE[:, ::1] forcevectors,
        list atoms,
        AINDEX n_atoms):

        cdef internal_atom *_atoms

        _atoms = <internal_atom*>malloc(
            len(atoms) * sizeof(internal_atom)
            )

        if _atoms == NULL:
            raise MemoryError()

        make_internal_atoms(atoms, _atoms)

        self._update(
            &structure[0, 0],
            &velocities[0, 0],
            &forcevectors[0, 0],
            _atoms,
            n_atoms,
            )

        if _atoms != NULL:
            free(_atoms)

    cdef void _update(
            self,
            AVALUE *structure,
            AVALUE *velocities,
            AVALUE *forcevectors,
            internal_atom *atoms,
            AINDEX n_atoms) nogil:

        cdef AINDEX index, d

        for index in range(n_atoms):
            for d in range(3):
                structure[index * 3 + d] = (
                    structure[index * 3 + d]
                    + velocities[index * 3 + d] * self._parameters[0]
                    + forcevectors[index * 3 + d] * 1.661e-12
                    * self._parameters[0]**2 / (2 * atoms[index].mass)
                    )
                velocities[index * 3 + d] = (
                    velocities[index * 3 + d]
                    + forcevectors[index * 3 + d]
                    * self._parameters[0] / atoms[index].mass
                    )
