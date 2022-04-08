from typing import Iterable, Mapping
from typing import Union

from cython.parallel cimport prange
import numpy as np

from eski.primitive_types import P_AINDEX, P_AVALUE


cdef Constants constants = make_constants()


cdef class Driver:
    """Base class for drivers to propagate

    If Cython supported abstract classes and/or virtual methods, this
    would be an abstract base class for the driver interface.  This
    class is not meant to be initialised.

    Args:
        parameters: Iterable of driver parameters
    """

    _param_names = []
    _resource_requirements = []
    _param_defaults = {}

    def __cinit__(self, parameters: list):

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
    def from_mapping(cls, parameters: Mapping[str, float] = None):
        if parameters is None:
            parameters = {}

        parameter_list = []
        for name in cls._param_names:
            value = parameters.get(name)
            if value is None:
                value = cls._param_defaults.get(name)
                if value is None:
                    raise ValueError(
                f"driver {cls.__name__!r} "
                f"requires parameter '{name}' "
                f"which was not given (no default provided)"
                )
            parameter_list.append(value)

        return cls(parameter_list)

    @property
    def parameters(self):
        pgenerator = (
            self._parameters[index]
            for index in range(self._n_parameters)
            )
        return dict(zip(self._param_names, pgenerator))

    def update(self, System system):
        self._update(system)

    cdef void _update(
            self, System system): ...

    def _check_param_consistency(self):
        cdef AINDEX dparam = len(self._param_names)

        if self._n_parameters != dparam:
            numerus_expect = "parameter" if dparam == 1 else "parameters"
            numerus_given = "was" if self._n_parameters == 1 else "were"

            raise ValueError(
                f"driver {type(self).__name__!r} "
                f"takes {dparam} {numerus_expect} "
                f"but {self._n_parameters} {numerus_given} given"
                )

    cdef void _on_startup(self, System system): ...


cdef class SteepestDescentMinimiser(Driver):
    """Local potential energy minimisation using a steepest-descent scheme
    """

    _param_names = ["tau", "tolerance", "tuneup", "tunedown"]
    _resource_requirements = ["configuration", "prev_epot"]
    _param_defaults = {
        "tau": 0.01,
        "tolerance": 100,
        "tuneup": 1.2,
        "tunedown": 0.2,
        }

    def __init__(self, *args, **kwargs):
        """The default constructor takes the following parameters in
        this exact order as a list of floats.

        Parameters:
            tau: step-size (nm).
            tolerance: convergence is reached when (kJ / (mol nm))
            tuneup: factor by which to increase tau on accepted step
            tunelow: factor by which to decrease tau on rejected step
        """
        self._check_param_consistency()

    cdef void _on_startup(self, System system):

        system._resources.configuration = np.zeros_like(
            system._configuration,
            dtype=P_AVALUE,
            order="c"
            )
        system._resources.prev_epot = system.potential_energy()
        self._adjusted_tau = self._parameters[0]

    cdef void _update(self, System system):

        cdef AINDEX index, d, i
        cdef AVALUE tolerance = self._parameters[1]
        cdef AVALUE tuneup = self._parameters[2]
        cdef AVALUE tunedown = self._parameters[3]
        cdef AVALUE *trial_configuration = &system._resources.configuration[0]
        cdef AVALUE max_f
        cdef AVALUE epot

        system.add_all_forces()

        with nogil:

            max_f = _get_max(system._forces_ptr, system._n_atoms * system._dim_per_atom)
            if max_f <= tolerance:
                system._stop = True
            else:
                for index in prange(system._n_atoms):
                    for d in range(system._dim_per_atom):
                        i = index * system._dim_per_atom + d
                        trial_configuration[i] = (
                            system._configuration[i]
                            + system._forces[i] / max_f * self._adjusted_tau
                            )

            system._configuration_ptr = trial_configuration

        epot = system.potential_energy()

        if epot < system._resources.prev_epot:
            self._adjusted_tau *= tuneup
            system._resources.prev_epot = epot

            with nogil:
                for i in prange(system._n_dim):
                    system._configuration[i] = trial_configuration[i]
        else:
            self._adjusted_tau *= tunedown

        system._configuration_ptr = &system._configuration[0]


cdef class EulerIntegrator(Driver):
    """Propagate positions and velocities with a forward Euler scheme

    The default constructor takes the following parameters in
    this exact order as a list of floats.

    Parameters:
        dt: Integration time step.
    """

    _param_names = ["dt"]
    _param_defaults = {
        "dt": 0.001
    }

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
                if atoms[index].mass <= 0:
                    continue

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
    _param_defaults = {
        "dt": 0.001,
        "friction": 1000,
        "T": 300
    }

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
                if atoms[index].mass <= 0:
                    continue

                sigma = csqrt(2 * constants.R * T / atoms[index].mass / friction)

                for d in range(dim_per_atom):
                    i = index * dim_per_atom + d
                    configuration[i] = (
                        configuration[i]
                        + forces[i] * dt / atoms[index].mass / friction
                        + sigma * _random_gaussian() * sqrt_dt
                        )
