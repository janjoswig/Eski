from typing import Iterable, Mapping
from typing import Union

from cython.parallel cimport prange
import numpy as np

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

    def update(
            self, System system):
        self._update(system)

    cdef void _update(
            self, System system): ...

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

    The default constructor takes the following parameters in
    this exact order as a list of floats.

    Parameters:
        dt: Integration time step.
    """

    _param_names = ["dt"]

    def __init__(self, *args, **kwargs):
        self._dparam = 1
        self._check_param_consistency()

    cdef void _update(self, System system):

        cdef AINDEX index, d, i
        cdef AVALUE dt = self._parameters[0]

        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef InternalAtom *atoms = system._atoms
        cdef AVALUE *configuration = &system._configuration[0]
        cdef AVALUE *velocities = &system._velocities[0]
        cdef AVALUE *forces = &system._forces[0]

        system.add_all_forces()

        with nogil:
            for index in prange(system._n_atoms):
                for d in range(dim_per_atom):
                    i = index * dim_per_atom + d
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
            System system):

        cdef AINDEX index, d, i
        cdef AVALUE dt = self._parameters[0]
        cdef AVALUE sqrt_dt = csqrt(dt)
        cdef AVALUE friction = self._parameters[1]
        cdef AVALUE T = self._parameters[2]
        cdef AVALUE sigma

        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef InternalAtom *atoms = system._atoms
        cdef AVALUE *configuration = &system._configuration[0]
        cdef AVALUE *forces = &system._forces[0]

        system.add_all_forces()

        with nogil:
            for index in prange(system._n_atoms):
                sigma = csqrt(2 * 0.008314463 * T / atoms[index].mass / friction)

                for d in range(dim_per_atom):
                    i = index * dim_per_atom + d
                    configuration[i] = (
                        configuration[i]
                        + forces[i] * dt / atoms[index].mass / friction
                        + sigma * _random_gaussian() * sqrt_dt
                        )
