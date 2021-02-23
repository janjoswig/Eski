from typing import Iterable, Mapping
from typing import Union

cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

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

        self.rv = np.zeros(3, dtype=P_AVALUE)
        self.fv = np.zeros(3, dtype=P_AVALUE)

    def __dealloc__(self):
        if self._indices != NULL:
            free(self._indices)

        if self._parameters != NULL:
            free(self._parameters)

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 0
        self._dindex = len(self._index_names)
        self._dparam = len(self._param_names)

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

    cpdef void add_contributions(
            self,
            AVALUE[:, ::1] structure,
            AVALUE[:, ::1] forcevectors):

        self._add_contributions(
                &structure[0, 0],
                &forcevectors[0, 0],
                )

    cdef void _add_contributions(
            self,
            AVALUE *structure,
            AVALUE *forcevectors) nogil:

        cdef AINDEX index

        for index in range(self._n_indices / self._dindex):
            self._add_contribution(
                index,
                structure,
                forcevectors,
                )

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors) nogil:
        NotImplemented


cdef class ForceHarmonicBond(Force):
    """Harmonic spring force approximating a chemical bond"""

    _index_names = ["p1", "p2"]
    _param_names = ["r0", "k"]

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 1
        self._dindex = len(self._index_names)
        self._dparam = len(self._param_names)

        self._check_index_param_consistency()

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors) nogil:
        """Evaluate harmonic bond force

        Args:
            index: Index of interaction
            structure: Pointer to atom position array
            forcevectors: Pointer to forces array

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
            &self.rv[0],
            &structure[p1 * 3],
            &structure[p2 * 3]
            )

        fv1 = &forcevectors[p1 * 3]
        fv2 = &forcevectors[p2 * 3]

        f = -k * (r - r0)
        for i in range(3):
            self.fv[i] = f * self.rv[i] / r
            fv1[i] += self.fv[i]
            fv2[i] -= self.fv[i]


cdef class ForceLJ(Force):
    """Harmonic spring force approximating a chemical bond"""

    _index_names = ["p1", "p2"]
    _param_names = ["sigma", "epsilon"]

    def __init__(self, *args, **kwargs):
        self.group = 0

        self._id = 2
        self._dindex = len(self._index_names)
        self._dparam = len(self._param_names)

        self._check_index_param_consistency()

    cdef void _add_contribution(
            self,
            AINDEX index,
            AVALUE *structure,
            AVALUE *forcevectors) nogil:
        """Evaluate Lennard-Jones force

        Args:
            index: Index of interaction
            structure: Pointer to atom position array
            forcevectors: Pointer to forces array

        Returns:
            Force (kJ / (mol nm))
        """

        cdef AINDEX i
        cdef AVALUE r, f
        cdef AINDEX p1 = self._indices[index * self._dindex]
        cdef AINDEX p2 = self._indices[index * self._dindex + 1]
        cdef AVALUE epsilon = self._parameters[index * self._dparam]
        cdef AVALUE sigma = self._parameters[index * self._dparam + 1]
        cdef AVALUE *fv1
        cdef AVALUE *fv2

        r = _euclidean_distance(
            &self.rv[0],
            &structure[p1 * 3],
            &structure[p2 * 3]
            )

        fv1 = &forcevectors[p1 * 3]
        fv2 = &forcevectors[p2 * 3]

        f = 24 * e * (2 * cpow(s, 12) / cpow(r, 13) - cpow(s, 6) / cpow(r, 7))
        for i in range(3):
            self.fv[i] = f * self.rv[i] / r
            fv1[i] += self.fv[i]
            fv2[i] -= self.fv[i]

    @staticmethod
    def lorentz_berthelot_combination(
            AVALUE s1, AVALUE e1, AVALUE s2, AVALUE e2):

        s = (s1 + s2) / 2
        e = csqrt(e1 * e2)

        return s, e