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
        indices: List of particle indices for which this force
            should be evaluated.
        parameters: List of force parameters.
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
            indices: Optional[list] = None,
            parameters: Optional[list] = None,
            *,
            group: int = 0,
            _id: Optional[int] = None,
            index_names: Optional[list] = None,
            param_names: Optional[list] = None,
            requires_gil: bool = False,
            **kwargs):

        cdef AINDEX i, index
        cdef AVALUE param

        if indices is not None:
            self._n_indices = len(indices)
            assert self._n_indices > 0, "Empty indices list"
            self._indices = self._allocate_and_fill_aindex_array(
                self._n_indices,
                indices)
        else:
            self._n_indices = 0

        if parameters is not None:
            self._n_parameters = len(parameters)
            assert self._n_parameters > 0, "Empty parameters list"
            self._parameters = self._allocate_and_fill_avalue_array(
                self._n_parameters,
                parameters)
        else:
            self._n_parameters = 0

        self.group = group
        self.requires_gil = requires_gil

    def __dealloc__(self):
        if self._indices != NULL:
            free(self._indices)

        if self._parameters != NULL:
            free(self._parameters)

    def __init__(
            self,
            indices: Optional[list] = None,
            parameters: Optional[list] = None,
            *,
            group: int = 0,
            _id: Optional[int] = None,
            index_names: Optional[list] = None,
            param_names: Optional[list] = None,
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

    cdef AVALUE* _allocate_and_fill_avalue_array(
            self, AINDEX n, list values):

        cdef AVALUE *ptr
        cdef AINDEX i

        ptr = <AVALUE*>malloc(n * sizeof(AVALUE))

        if ptr == NULL:
            raise MemoryError()

        for i in range(n):
            ptr[i] = values[i]

        return ptr

    cdef AINDEX* _allocate_and_fill_aindex_array(
            self, AINDEX n, list values):

        cdef AINDEX *ptr
        cdef AINDEX i

        ptr = <AINDEX*>malloc(n * sizeof(AINDEX))

        if ptr == NULL:
            raise MemoryError()

        for i in range(n):
            ptr[i] = values[i]

        return ptr

    cpdef void _check_index_param_consistency(self) except *:
        """Raise error if indices and parameters do not match"""

        if (self._n_indices % self._dindex) > 0:
            raise ValueError(
                f"Wrong number of 'indices'; must be multiple of {self._dindex}"
                )

        if self._dparam == 0:
            if self._n_parameters == 0:
                return
            raise ValueError(
                f"Force {type(self).__name__!r} takes no parameters"
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

    cdef void _add_all_forces(self, System system):
        cdef AINDEX index

        for index in range(self._n_indices // self._dindex):
            self._add_force_by_index(index, system)

    cdef void _add_all_forces_nogil(self, System system) nogil:

        cdef AINDEX index

        for index in range(self._n_indices // self._dindex):
            self._add_force_by_index_nogil(index, system)

    def add_force(self, indices: list, parameters: list, system):
        cdef AINDEX i

        cdef AINDEX* indices_ptr = self._allocate_and_fill_aindex_array(
            len(indices), indices
            )

        cdef AVALUE* parameters_ptr = self._allocate_and_fill_avalue_array(
            len(parameters), parameters
            )

        if self.requires_gil:
            self._add_force(indices_ptr, parameters_ptr, system)
        else:
            self._add_force_nogil(indices_ptr, parameters_ptr, system)

        free(indices_ptr)
        free(parameters_ptr)

    cdef void _add_force(
        self,
        AINDEX *indices,
        AVALUE *parameters,
        System system): ...

    cdef void _add_force_nogil(
        self,
        AINDEX *indices,
        AVALUE *parameters,
        System system) nogil: ...

    cdef void _add_force_by_index(
            self,
            AINDEX index,
            System system):
        """Evaluate force with indices and parameters from interaction index

        Note:
            Calls implementation :func:`~eski.md.Interaction._add_force`

        Args:
            index: Index of interaction
            system: Instance of :class:`eski.md.System`
        """

        self._add_force(
            &self._indices[index * self._dindex],
            &self._parameters[index * self._dparam],
            system
            )

    cdef void _add_force_by_index_nogil(
            self,
            AINDEX index,
            System system) nogil:
        """Evaluate force with indices and parameters from interaction index

        Note:
            Calls nogil implementation :func:`~eski.md.Interaction._add_force_nogil`

        Args:
            index: Index of interaction
            system: Instance of :class:`eski.md.System`
        """

        self._add_force_nogil(
            &self._indices[index * self._dindex],
            &self._parameters[index * self._dparam],
            system
            )

    def  get_energy(self, indices: list, parameters: list, system):
        cdef AINDEX i
        cdef AVALUE energy

        cdef AINDEX* indices_ptr = self._allocate_and_fill_aindex_array(
            len(indices), indices
            )

        cdef AVALUE* parameters_ptr = self._allocate_and_fill_avalue_array(
            len(parameters), parameters
            )

        if self.requires_gil:
            energy = self._get_energy(indices_ptr, parameters_ptr, system)
        else:
            energy = self._get_energy_nogil(indices_ptr, parameters_ptr, system)

        free(indices_ptr)
        free(parameters_ptr)

        return energy

    cdef AVALUE _get_energy(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system): ...

    cdef AVALUE _get_energy_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil: ...

    cdef AVALUE _get_total_energy(
            self,  System system):

        cdef AINDEX index
        cdef AVALUE energy = 0

        for index in range(self._n_indices / self._dindex):
            energy = energy + self._get_energy_by_index(index, system)

        return energy

    cdef AVALUE _get_total_energy_nogil(
            self,  System system) nogil:

        cdef AINDEX index
        cdef AVALUE energy = 0

        for index in range(self._n_indices / self._dindex):
            energy = energy + self._get_energy_by_index_nogil(index, system)

        return energy

    cdef AVALUE _get_energy_by_index(
            self,
            AINDEX index,
            System system):

        return self._get_energy(
            &self._indices[index * self._dindex],
            &self._parameters[index * self._dparam],
            system
            )

    cdef AVALUE _get_energy_by_index_nogil(
            self,
            AINDEX index,
            System system) nogil:

        return self._get_energy_nogil(
            &self._indices[index * self._dindex],
            &self._parameters[index * self._dparam],
            system
            )


cdef class ConstantBias(Interaction):
    """Constant force on single atoms"""

    _default_index_names = ["p1"]
    _default_param_names = []
    _default_id = 10

    def __init__(self, *args, **kwargs):
        """Initialise constant bias force

        Note:
            A list of parameter names needs to be specified to
            state in how many dimensions the force should be applied

        Args:
            indices: list of individual atoms to which a constant force should be
                applied
            parameters: list of forces in each dimension
        """
        if kwargs["param_names"] is None:
            raise ValueError(
                "This interaction type requires `param_names`"
                )

        super().__init__(*args, **kwargs)

    cdef void _add_force_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        cdef AINDEX i
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]

        for i in range(d):
            f1[i] += parameters[i]

    cdef AVALUE _get_energy_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        return 0


cdef class Exclusion(Interaction):
    """Removes forces acting on atoms"""

    _default_index_names = ["p1"]
    _default_param_names = []
    _default_id = 10

    cdef void _add_force_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        cdef AINDEX i
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]

        for i in range(d):
            f1[i] = 0

    cdef AVALUE _get_energy_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        return 0


cdef class HarmonicPositionRestraint(Interaction):
    """Harmonic spring force to restrain atom positions"""

    _default_index_names = ["p1"]
    _default_param_names = []
    _default_id = 1

    def __init__(self, *args, **kwargs):
        if kwargs["param_names"] is None:
            raise ValueError(
                "This interaction type requires `param_names`"
                )

        super().__init__(*args, **kwargs)

    cdef void _add_force_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        cdef AINDEX i
        cdef AVALUE r, f
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE r0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *anchor = &parameters[2]
        cdef AVALUE *rv = system._resources.rva

        system._pbc._pbc_distance(rv, c1, anchor, d)
        r = _norm2(rv, d)

        if r != 0:
            f = -k * (r - r0)
            for i in range(d):
                f1[i] += f * rv[i] / r

    cdef AVALUE _get_energy_nogil(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        return 0


cdef class HarmonicBond(Interaction):
    """Harmonic spring force approximating a chemical bond"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["r0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects two particles with parameters r0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE r, f, _f
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE r0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rv = system._resources.rva

        system._pbc._pbc_distance(rv, c1, c2, d)
        r = _norm2(rv, d)

        f = -k * (r - r0)
        for i in range(d):
            _f = f * rv[i] / r
            f1[i] += _f
            f2[i] -= _f

    cdef AVALUE _get_energy_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE r
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE r0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rv = system._resources.rva

        system._pbc._pbc_distance(rv, c1, c2, d)
        r = _norm2(rv, d)

        return 0.5 * k * cpow(r - r0, 2)


cdef class HarmonicAngle(Interaction):
    """Harmonic force approximating a valence angle"""

    _default_index_names = ["p1", "p2", "p3"]
    _default_param_names = ["theta0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects three particles with parameters theta0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE ra, rb, f
        cdef AVALUE cos_theta, sin_inv
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE *f3 = &system._forces_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rva = system._resources.rva
        cdef AVALUE *rvb = system._resources.rvb
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system._pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system._pbc._pbc_distance(rvb, c3, c2, d)
        rb = _norm2(rvb, d)

        cos_theta = 0
        for i in range(d):
            cos_theta = cos_theta + (rva[i] / ra) * ( rvb[i] / rb)

        sin_inv = 1.0 / csqrt(1.0 - cos_theta * cos_theta)

        for i in range(d):
            der1[i] = sin_inv * (cos_theta  * (rva[i] / ra) - (rvb[i] / rb)) / ra
            der3[i] = sin_inv * (cos_theta  * (rvb[i] / rb) - (rva[i] / ra)) / rb
            der2[i] = -(der1[i] + der3[i])

        f = -k * (cacos(cos_theta) - theta0)
        for i in range(d):
            f1[i] += f * der1[i]
            f2[i] += f * der2[i]
            f3[i] += f * der3[i]

    cdef AVALUE _get_energy_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE ra, rb
        cdef AVALUE cos_theta
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rva = system._resources.rva
        cdef AVALUE *rvb = system._resources.rvb
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system._pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system._pbc._pbc_distance(rvb, c3, c2, d)
        rb = _norm2(rvb, d)

        cos_theta = 0
        for i in range(d):
            cos_theta = cos_theta + (rva[i] / ra) * ( rvb[i] / rb)

        return 0.5 * k * cpow(cacos(cos_theta) - theta0, 2)


cdef class CosineHarmonicAngle(Interaction):
    """Harmonic force approximating a valence angle"""

    _default_index_names = ["p1", "p2", "p3"]
    _default_param_names = ["costheta0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects three particles with parameters theta0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE ra, rb, f
        cdef AVALUE cos_theta, sin_inv
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE *f3 = &system._forces_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rva = system._resources.rva
        cdef AVALUE *rvb = system._resources.rvb
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system._pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system._pbc._pbc_distance(rvb, c3, c2, d)
        rb = _norm2(rvb, d)

        cos_theta = 0
        for i in range(d):
            cos_theta = cos_theta + (rva[i] / ra) * ( rvb[i] / rb)

        sin_inv = 1.0 / csqrt(1.0 - cos_theta * cos_theta)

        for i in range(d):
            der1[i] = sin_inv * (cos_theta  * (rva[i] / ra) - (rvb[i] / rb)) / ra
            der3[i] = sin_inv * (cos_theta  * (rvb[i] / rb) - (rva[i] / ra)) / rb
            der2[i] = -(der1[i] + der3[i])

        f = k * (cos_theta - theta0) * csin(cacos(cos_theta))
        for i in range(d):
            f1[i] += f * der1[i]
            f2[i] += f * der2[i]
            f3[i] += f * der3[i]

    cdef AVALUE _get_energy_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE ra, rb
        cdef AVALUE cos_theta
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rva = system._resources.rva
        cdef AVALUE *rvb = system._resources.rvb
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system._pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system._pbc._pbc_distance(rvb, c3, c2, d)
        rb = _norm2(rvb, d)

        cos_theta = 0
        for i in range(d):
            cos_theta = cos_theta + (rva[i] / ra) * ( rvb[i] / rb)

        return 0.5 * k * cpow(cos_theta - theta0, 2)


cdef class LJ(Interaction):
    """Non-bonded Lennard-Jones interaction"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["sigma", "epsilon"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    cdef void _add_force_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE r, f, _f
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE s = parameters[0]
        cdef AVALUE e = parameters[1]
        cdef AVALUE *rv = system._resources.rva

        system._pbc._pbc_distance(rv, c1, c2, d)
        r = _norm2(rv, d)

        f = 24 * e / r * (2 * cpow(s / r, 12) - cpow(s / r, 6))
        for i in range(d):
            _f = f * rv[i] / r
            f1[i] += _f
            f2[i] -= _f

    cdef AVALUE _get_energy_nogil(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE r, f
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE s = parameters[0]
        cdef AVALUE e = parameters[1]
        cdef AVALUE *rv = system._resources.rva

        system._pbc._pbc_distance(rv, c1, c2, d)
        r = _norm2(rv, d)

        return 4 * e * (cpow(s / r, 12) - cpow(s / r, 6))

    @staticmethod
    def lorentz_berthelot_combination(
            AVALUE s1, AVALUE e1, AVALUE s2, AVALUE e2):

        s = (s1 + s2) / 2
        e = csqrt(e1 * e2)

        return s, e


cdef class CoulombPME(Interaction):
    """
    Smeared out Gauss (alpha/sqrt(pi))**3 exp(-alpha**2 r**2)
    """
    pass