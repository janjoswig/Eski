from typing import Iterable, Mapping
from typing import Union

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from eski.md cimport System
from eski.primitive_types import P_AINDEX, P_AVALUE


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

    cdef void update(self, System system):
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
