from abc import ABC, abstractmethod
from typing import Iterable, Mapping
from typing import Optional, Union

cimport cython
import numpy as np

from eski.primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class Interaction:
    """Base class for interaction to evaluate

    If Cython supported abstract classes and/or virtual methods, this
    would be an abstract base class for the interaction interface.
    This class is not meant to be initialised.

    Args:
        indices: Iterable of particle indices for which this force
            should be evaluated.
        parameters: Iterable of force parameters.
        group: Force group. Useful to distinguish between forces
            that should be evaluated at different times steps.
        _id: Unique ID of this force type.
        index_names: List of index identifiers.
        param_names: List of parameter identifiers
    """

    _default_index_names = ["p1"]
    _default_param_names = ["x"]
    _default_id = 0

    def __cinit__(
            self,
            indices: Iterable[int],
            parameters: Iterable[float],
            *,
            group: int = 0,
            _id: Optional[int] = None,
            index_names: Optional[Iterable[str]] = None,
            param_names: Optional[Iterable[str]] = None,
            **kwargs):

        cdef AINDEX i, index
        cdef AVALUE param

        self._n_indices = len(indices)
        self._n_parameters = len(parameters)

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

        self.group = group

    def __dealloc__(self):
        if self._indices != NULL:
            free(self._indices)

        if self._parameters != NULL:
            free(self._parameters)

    def __init__(
            self,
            indices: Iterable[int],
            parameters: Iterable[float],
            *,
            group: int = 0,
            _id: Optional[int] = None,
            index_names: Optional[Iterable[str]] = None,
            param_names: Optional[Iterable[str]] = None,
            **kwargs):

        if index_names is None:
            index_names = self._default_index_names
        self._index_names = index_names

        if param_names is None:
            param_names = self._default_param_names
        self._param_names = param_names

        self._dindex = len(self._index_names)
        self._dparam = len(self._param_names)

        self._check_index_param_consistency()

        if _id is None:
            _id = self._default_id
        self._id = _id

    def __repr__(self):
        attr_repr = ", ".join(
            [
                f"group={self.group}",
                f"n_interactions={self.n_interactions}"
            ]
        )
        return f"{self.__class__.__name__}({attr_repr})"

    @property
    def id(self):
       return self._id

    @property
    def n_interactions(self):
        return self._n_indices // self._dindex

    @classmethod
    def from_mappings(
            cls,
            interactions: Iterable[Mapping[str, Union[float, int]]],
            group=0, _id=None,
            index_names=None, param_names=None, **kwargs):

        if index_names is None:
            index_names = cls._default_index_names

        if param_names is None:
            param_names = cls._default_param_names

        indices = []
        parameters = []
        for mapping in interactions:
            for name in index_names:
                indices.append(mapping[name])

            for name in param_names:
                parameters.append(mapping[name])

        return cls(
            indices, parameters,
            group=group, _id=_id,
            index_names=index_names,
            param_names=param_names,
            **kwargs
            )

    cpdef void _check_index_param_consistency(self) except *:
        """Raise error if indices and parameters do not match"""

        if cython.cmod(self._n_indices, self._dindex) > 0:
            raise ValueError(
                f"Wrong number of 'indices'; must be multiple of {self._dindex}"
                )

        if self._dparam == 0:
            if self._n_parameters == 0:
                return
            raise ValueError(
                f"Force {type(self).__name__!r} takes no parameters"
                )

        if cython.cmod(self._n_parameters, self._dparam) > 0:
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
        """Return info for interaction

        Args:
            index: Index of the interaction to get the info for

        Returns:
            Dictionary with keys according to
            :obj:`self._index_names` and :obj:`self._param_names` and
            corresponding values
        """

        self._check_interaction_index(index)

        cdef dict info = {}
        cdef AINDEX i
        cdef str name

        for i, name in enumerate(self._index_names):
            info[name] = self._indices[index * self._dindex + i]

        for i, name in enumerate(self._param_names):
            info[name] = self._parameters[index * self._dparam + i]

        return info

    cpdef void _check_interaction_index(self, AINDEX index) except *:
        if (index < 0) or (index >= self.n_interactions):
            raise IndexError(
                "Interaction index out of range"
                )

    cpdef void add_all_forces(
            self,
            System system):

        cdef AINDEX index

        for index in range(self._n_indices // self._dindex):
            self.add_force_by_index(
                index,
                system
                )

    cpdef void add_force_by_index(
            self,
            AINDEX index,
            System system): ...

    cdef void _add_all_forces(
            self,
            System system) nogil:

        cdef AINDEX index

        for index in range(self._n_indices // self._dindex):
            self._add_force_by_index(
                index,
                system
                )

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            System system) nogil: ...

    cpdef AVALUE get_total_energy(
            self,  System system):

        cdef AINDEX index
        cdef AVALUE energy = 0

        for index in range(self._n_indices / self._dindex):
            energy = energy + self._get_energy_by_index(
                index,
                system
                )

        return energy

    cpdef AVALUE get_energy_by_index(
            self,
            AINDEX index,
            System system): ...

    cdef AVALUE _get_total_energy(
            self,  System system) nogil:

        cdef AINDEX index
        cdef AVALUE energy = 0

        for index in range(self._n_indices / self._dindex):
            energy = energy + self._get_energy_by_index(
                index,
                system
                )

        return energy

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            System system) nogil: ...


cdef class ConstantBias(Interaction):
    """Constant force applied to a single atom in each dimension

    On initialisation a list of parameter names should be given
    that matches in length the number of dimensions per atom.
    """

    _default_index_names = ["p1"]
    _default_param_names = []
    _default_id = 10

    def __init__(self, *args, **kwargs):
        if kwargs["param_names"] is None:
            raise ValueError(
                "This interaction type requires `param_names`"
                )

        super().__init__(*args, **kwargs)

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            System system) nogil:
        """Evaluate biasing force

        Args:
            index: Index of interaction
            configuration: Pointer to atom position array
            forces: Pointer to forces array.
                Force in (kJ / (mol nm)).
        """

        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AVALUE *fv1
        cdef AVALUE *b
        cdef AVALUE *forces = &system._forces[0]

        fv1 = &forces[p1 * system._dim_per_atom]
        b = &self._parameters[index * self._dparam]

        for i in range(system._dim_per_atom):
            fv1[i] += b[i]

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            System system) nogil:

        return 0


cdef class HarmonicBond(Interaction):
    """Harmonic spring force approximating a chemical bond"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["r0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            System system) nogil:
        """Evaluate harmonic bond force

        Args:
            index: Index of interaction
            system: :class:`eski.md.System`
        """

        cdef AINDEX i
        cdef AVALUE r, f, _f
        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AINDEX p2 = self._indices[index * self._dindex + 1]
        cdef AVALUE r0 = self._parameters[index * self._dparam]
        cdef AVALUE k = self._parameters[index * self._dparam + 1]
        cdef AVALUE *fv1
        cdef AVALUE *fv2
        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef AVALUE *configuration = &system._configuration[0]
        cdef AVALUE *forces = &system._forces[0]


        r = _euclidean_distance(
            &system._resources.rv[0],
            &configuration[p1 * dim_per_atom],
            &configuration[p2 * dim_per_atom],
            system._dim_per_atom
            )

        fv1 = &forces[p1 * dim_per_atom]
        fv2 = &forces[p2 * dim_per_atom]

        f = -k * (r - r0)
        for i in range(dim_per_atom):
            _f = f * system._resources.rv[i] / r
            fv1[i] += _f
            fv2[i] -= _f

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            System system) nogil:

        cdef AVALUE r
        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AINDEX p2 = self._indices[index * self._dindex + 1]
        cdef AVALUE r0 = self._parameters[index * self._dparam]
        cdef AVALUE k = self._parameters[index * self._dparam + 1]
        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef AVALUE *configuration = &system._configuration[0]

        r = _euclidean_distance(
            &system._resources.rv[0],
            &configuration[p1 * dim_per_atom],
            &configuration[p2 * dim_per_atom],
            dim_per_atom
            )

        return 0.5 * k * cpow(r - r0, 2)


cdef class LJ(Interaction):
    """Harmonic spring force approximating a chemical bond"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["sigma", "epsilon"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            System system) nogil:
        """Evaluate Lennard-Jones force

        Args:
            index: Index of interaction
            system: :class:`eski.md.System`
        """

        cdef AINDEX i
        cdef AVALUE r, f, _f
        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AINDEX p2 = self._indices[index * self._dindex + 1]
        cdef AVALUE e = self._parameters[index * self._dparam]
        cdef AVALUE s = self._parameters[index * self._dparam + 1]
        cdef AVALUE *fv1
        cdef AVALUE *fv2
        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef AVALUE *configuration = &system._configuration[0]
        cdef AVALUE *forces = &system._forces[0]

        r = _euclidean_distance(
            &system._resources.rv[0],
            &configuration[p1 * dim_per_atom],
            &configuration[p2 * dim_per_atom],
            dim_per_atom
            )

        fv1 = &forces[p1 * dim_per_atom]
        fv2 = &forces[p2 * dim_per_atom]

        f = 24 * e * (2 * cpow(s, 12) / cpow(r, 13) - cpow(s, 6) / cpow(r, 7))
        for i in range(dim_per_atom):
            _f = f * system._resources.rv[i] / r
            fv1[i] += _f
            fv2[i] -= _f

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            System system) nogil:

        cdef AVALUE r
        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AINDEX p2 = self._indices[index * self._dindex + 1]
        cdef AVALUE e = self._parameters[index * self._dparam]
        cdef AVALUE s = self._parameters[index * self._dparam + 1]
        cdef AINDEX dim_per_atom = system._dim_per_atom
        cdef AVALUE *configuration = &system._configuration[0]

        r = _euclidean_distance(
            &system._resources.rv[0],
            &configuration[p1 * dim_per_atom],
            &configuration[p2 * dim_per_atom],
            dim_per_atom
            )

        return 4 * e * (cpow(s / r, 12) - cpow(s / r, 6))

    @staticmethod
    def lorentz_berthelot_combination(
            AVALUE s1, AVALUE e1, AVALUE s2, AVALUE e2):

        s = (s1 + s2) / 2
        e = csqrt(e1 * e2)

        return s, e
