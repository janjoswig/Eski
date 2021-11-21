from typing import Iterable, Mapping
from typing import Union

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as csqrt, pow as cpow, log as clog

from eski.primitive_types import P_AINDEX, P_AVALUE
from eski.atoms cimport internal_atom, make_internal_atoms
from eski.metrics cimport _random_gaussian


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
        AVALUE[::1] configuration,
        AVALUE[::1] velocities,
        AVALUE[::1] forces,
        list atoms,
        system_support support):

        cdef internal_atom *_atoms

        _atoms = <internal_atom*>malloc(
            len(atoms) * sizeof(internal_atom)
            )

        if _atoms == NULL:
            raise MemoryError()

        make_internal_atoms(atoms, _atoms)

        self._update(
            &configuration[0],
            &velocities[0],
            &forces[0],
            &_atoms[0],
            support,
            )

        if _atoms != NULL:
            free(_atoms)

    cdef void _update(
            self,
            AVALUE *configuration,
            AVALUE *velocities,
            AVALUE *forces,
            internal_atom *atoms,
            system_support support) nogil:
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
    """Propagate positions and velocities with a forward Euler scheme

    Parameters:
        dt:
    """

    _param_names = ["dt"]

    def __init__(self, *args, **kwargs):
        self._dparam = 1
        self._check_param_consistency()

    cdef void _update(
            self,
            AVALUE *configuration,
            AVALUE *velocities,
            AVALUE *forces,
            internal_atom *atoms,
            system_support support) nogil:

        cdef AINDEX index, d, i
        cdef AVALUE dt = self._parameters[0]

        for index in range(support.n_atoms):
            for d in range(support.dim_per_atom):
                i = index * support.dim_per_atom + d
                configuration[i] = (
                    configuration[i]
                    + velocities[i] * dt
                    + forces[i] * dt**2 / (2 * atoms[index].mass)
                    )
                velocities[i] = (
                    velocities[i]
                    + forces[i]
                    * dt / atoms[index].mass
                    )


cdef class EulerMaruyamaIntegrator(Driver):
    """Propagate positions and velocities with a Euler-Maruyama scheme

    Parameters:
        dt:
        friction: gamma
        T:
    """

    _param_names = ["dt", "friction", "T"]

    def __init__(self, *args, **kwargs):
        self._dparam = 3
        self._check_param_consistency()

    cdef void _update(
            self,
            AVALUE *configuration,
            AVALUE *velocities,
            AVALUE *forces,
            internal_atom *atoms,
            system_support support) nogil:

        cdef AINDEX index, d, i
        cdef AVALUE dt = self._parameters[0]
        cdef AVALUE friction = self._parameters[1]
        cdef AVALUE T = self._parameters[2]
        cdef AVALUE sigma

        for index in range(support.n_atoms):
            sigma = csqrt(2 * 0.008314463 * T / atoms[index].mass / friction)

            for d in range(support.dim_per_atom):
                i = index * support.dim_per_atom + d
                configuration[i] = (
                    configuration[i]
                    + forces[i] * dt / atoms[index].mass / friction
                    + sigma * _random_gaussian() * csqrt(dt)
                    )
