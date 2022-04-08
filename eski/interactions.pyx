from abc import ABC, abstractmethod
from typing import Iterable, Mapping
from typing import Optional, Union

cimport cython
import numpy as np

from eski.primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class InteractionProvider:

    cdef IVPTRPAIR get_interaction_by_index(self, AINDEX index, Interaction interaction) nogil: ...
    cpdef void _check_index_param_consistency(self, Interaction interaction) except *: ...
    cpdef void _check_interaction_index(self, AINDEX index, Interaction interaction) except *: ...
    cdef inline AINDEX _n_interactions(self, Interaction interaction) nogil:
        return self._n_indices // interaction._dindex

    def n_interactions(self, Interaction interaction):
        return self._n_interactions(interaction)

cdef class NoProvider(InteractionProvider):
    pass

cdef class NeighboursProvider(InteractionProvider):
    pass

cdef class ExplicitProvider(InteractionProvider):
    def __cinit__(
            self,
            indices: list,
            parameters: list,
            *,
            **kwargs):

        assert len(indices) > 0, "Empty indices list"
        assert len(parameters) > 0, "Empty parameters list"

        self._n_indices = len(indices)
        self._indices = _allocate_and_fill_aindex_array(
            self._n_indices,
            indices
            )

        self._n_parameters = len(parameters)
        self._parameters = _allocate_and_fill_avalue_array(
            self._n_parameters,
            parameters
            )

    def __dealloc__(self):
        if self._indices != NULL:
            free(self._indices)

        if self._parameters != NULL:
            free(self._parameters)

    cpdef void _check_index_param_consistency(self, Interaction interaction) except *:
        """Raise error if indices and parameters do not match"""

        if (self._n_indices % interaction._dindex) > 0:
            raise ValueError(
                f"Wrong number of 'indices'; must be multiple of {interaction._dindex}"
                )

        if interaction._dparam == 0:
            if self._n_parameters == 0:
                return
            raise ValueError(
                f"Force {type(self).__name__!r} takes no parameters"
                )

        if (self._n_parameters % interaction._dparam) > 0:
            raise ValueError(
                f"Wrong number of 'parameters'; must be multiple of {interaction._dparam}"
                )

        len_no_match = (
            (self._n_indices / interaction._dindex) !=
            (self._n_parameters / interaction._dparam)
        )
        if len_no_match:
            raise ValueError(
                "Length of 'indices' and 'parameters' does not match"
                )

    cpdef void _check_interaction_index(
            self, AINDEX index, Interaction interaction) except *:
        if (index < 0) or (index >= self._n_interactions(interaction)):
            raise IndexError(
                "Interaction index out of range"
                )

    def get_interaction(self, AINDEX index, Interaction interaction):
        """Return info for interaction

        Args:
            index: Index of the interaction to get the info for
            interaction: Interaction type

        Returns:
            Dictionary with keys according to
            :obj:`self._index_names` and :obj:`self._param_names` and
            corresponding values
        """

        self._check_interaction_index(index, interaction)

        cdef dict info = {}
        cdef AINDEX i
        cdef str name

        for i, name in enumerate(interaction._index_names):
            info[name] = self._indices[index * interaction._dindex + i]

        for i, name in enumerate(interaction._param_names):
            info[name] = self._parameters[index * interaction._dparam + i]

        return info

    cdef IVPTRPAIR get_interaction_by_index(
            self, AINDEX index, Interaction interaction) nogil:

        cdef IVPTRPAIR ivpair

        ivpair = make_pair(
            <AINDEX*>&self._indices[index * interaction._dindex],
            <AVALUE*>&self._parameters[index * interaction._dparam]
            )

        return ivpair


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
    _default_requires_gil = False

    def __cinit__(
            self,
            provider=None,
            *,
            group: int = 0,
            _id: Optional[int] = None,
            index_names: Optional[list] = None,
            param_names: Optional[list] = None,
            **kwargs):

        if provider is None:
            self.provider = NoProvider()

        self.group = group

    def __init__(
            self,
            provider=None,
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

        self.provider = provider
        self.provider._check_index_param_consistency(self)

        if _id is None:
            _id = self._default_id
        self._id = _id

        self.requires_gil = self._default_requires_gil

    def __repr__(self):
        attr_repr = ", ".join(
            [
                f"group={self.group}",
            ]
        )
        return f"{self.__class__.__name__}({attr_repr})"

    @classmethod
    def from_explicit(
        cls,
        indices: list, parameters: list, *,
        provider_kwargs=None, **kwargs):

        if provider_kwargs is None:
            provider_kwargs = {}

        return cls(
            ExplicitProvider(indices, parameters, **provider_kwargs),
            **kwargs
            )

    @property
    def id(self):
       return self._id

    cdef void _add_all_forces(self, System system):
        cdef AINDEX index
        cdef IVPTRPAIR ivpair

        for index in range(self.provider._n_interactions(self)):
            ivpair = self.provider.get_interaction_by_index(index, self)
            self._add_force(ivpair.first, ivpair.second, system)

    cdef void _add_all_forces_nogil(self, System system) nogil:
        cdef AINDEX index
        cdef IVPTRPAIR ivpair

        for index in range(self.provider._n_interactions(self)):
            ivpair = self.provider.get_interaction_by_index(index, self)
            self._add_force_nogil(ivpair.first, ivpair.second, system)

    def add_force(self, indices: list, parameters: list, System system):
        cdef AINDEX i

        cdef AINDEX* indices_ptr = _allocate_and_fill_aindex_array(
            len(indices), indices
            )

        cdef AVALUE* parameters_ptr = _allocate_and_fill_avalue_array(
            len(parameters), parameters
            )

        if self.requires_gil:
            self._add_force(indices_ptr, parameters_ptr, system)
        else:
            with nogil: self._add_force_nogil(indices_ptr, parameters_ptr, system)

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

    def  get_energy(self, indices: list, parameters: list, system):
        cdef AINDEX i
        cdef AVALUE energy

        cdef AINDEX* indices_ptr = _allocate_and_fill_aindex_array(
            len(indices), indices
            )

        cdef AVALUE* parameters_ptr = _allocate_and_fill_avalue_array(
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
        cdef IVPTRPAIR ivpair
        cdef AVALUE energy = 0

        for index in range(self.provider._n_interactions(self)):
            ivpair = self.provider.get_interaction_by_index(index, self)
            energy = energy + self._get_energy(ivpair.first, ivpair.second, system)

        return energy

    cdef AVALUE _get_total_energy_nogil(
            self,  System system) nogil:

        cdef AINDEX index
        cdef IVPTRPAIR ivpair
        cdef AVALUE energy = 0

        for index in range(self.provider._n_interactions(self)):
            ivpair = self.provider.get_interaction_by_index(index, self)
            energy = energy + self._get_energy_nogil(ivpair.first, ivpair.second, system)

        return energy


cdef class CustomInteraction(Interaction):
    _default_requires_gil = True


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

        system.pbc._pbc_distance(rv, c1, anchor, d)
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

        system.pbc._pbc_distance(rv, c1, c2, d)
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

        system.pbc._pbc_distance(rv, c1, c2, d)
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

        system.pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system.pbc._pbc_distance(rvb, c3, c2, d)
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

        system.pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system.pbc._pbc_distance(rvb, c3, c2, d)
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

        system.pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system.pbc._pbc_distance(rvb, c3, c2, d)
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

        system.pbc._pbc_distance(rva, c1, c2, d)
        ra = _norm2(rva, d)
        system.pbc._pbc_distance(rvb, c3, c2, d)
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

        system.pbc._pbc_distance(rv, c1, c2, d)
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

        system.pbc._pbc_distance(rv, c1, c2, d)
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