from typing import Iterable, Mapping
from typing import Union

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from eski.md cimport System
from eski.metrics cimport _euclidean_distance
from eski.primitive_types import P_AINDEX, P_AVALUE


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

    _index_names = ["p1"]
    _param_names = []

    def __cinit__(
            self,
            indices: Iterable[int],
            parameters: Iterable[float]):

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

    def __dealloc__(self):
        if self._indices != NULL:
            free(self._indices)

        if self._parameters != NULL:
            free(self._parameters)

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 0
        self._dindex = 1
        self._dparam = 0

        self._check_index_param_consistency()

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
        return self._n_indices / self._dindex

    @classmethod
    def from_mappings(cls, forces: Iterable[Mapping[str, Union[float, int]]]):
        indices = []
        parameters = []
        for mapping in forces:
            for name in cls._index_names:
                indices.append(mapping[name])

            for name in cls._param_names:
                parameters.append(mapping[name])

        return cls(indices, parameters)

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
        """Return info for interaction

        Args:
            index: Index of the interaction to get the info for

        Returns:
            Dictionary with keys according to
            :obj:`self._index_names` and :obj:`self._param_names` and
            corresponding values
        """

        self._check_interaction_index(index)

        info = {}
        for i, name in enumerate(self._index_names):
            info[name] = self._indices[index * self._dindex + i]

        for i, name in enumerate(self._param_names):
            info[name] = self._parameters[index * self._dparam + i]

        return info

    def _check_interaction_index(self, AINDEX index):
        if (index < 0) or (index >= self.n_interactions):
            raise IndexError(
                "Interaction index out of range"
                )

    cpdef void add_contributions(self, System system):
        NotImplemented

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors,
            AVALUE *rv,
            AVALUE *fv) nogil:
        NotImplemented


cdef class ForceHarmonicBond(Force):
    """Harmonic spring force approximating a chemical bond"""

    _index_names = ["p1", "p2"]
    _param_names = ["r0", "k"]

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 1
        self._dindex = 2
        self._dparam = 2

        self._check_index_param_consistency()

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
            forcevectors: Pointer to forces array
            rv: Pointer to buffer array of length 3 for distance vector
            fv: Pointer to buffer array of length 3 for force vector

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
