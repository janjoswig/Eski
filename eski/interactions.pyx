from abc import ABC, abstractmethod
from typing import Iterable, Mapping
from typing import Optional, Union

from cython.parallel cimport prange
import numpy as np

from eski.primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class InteractionProvider:

    cpdef void _check_consistency(self) except *: ...
    cdef (AINDEX*, AVALUE*) _next_interaction(self) nogil: ...
    cdef void _reset_iteration(self) nogil: ...

    def __init__(self, *args, **kwargs):
        if type(self) is InteractionProvider:
            raise RuntimeError(
                f"Cannot instantiate abstract class {type(self)}"
                )

cdef class NoProvider(InteractionProvider):
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

    cpdef void _check_consistency(self) except *:
        """Raise error if indices and parameters do not match"""

        cdef AINDEX n_interactions

        if (self._n_indices % self._interaction._dindex) > 0:
            raise ValueError(
                f"Wrong number of 'indices'; must be multiple of {self._interaction._dindex}"
                )

        if self._interaction._dparam == 0:
            if self._n_parameters == 0:
                return
            raise ValueError(
                f"Force {type(self).__name__!r} takes no parameters"
                )

        if (self._n_parameters % self._interaction._dparam) > 0:
            raise ValueError(
                f"Wrong number of 'parameters': "
                f"must be multiple of {self._interaction._dparam}"
                )

        n_interactions = self._n_indices // self._interaction._dindex

        len_no_match = (
            n_interactions != (self._n_parameters // self._interaction._dparam)
        )
        if len_no_match:
            raise ValueError(
                f"Length of 'indices' ({self._n_indices}) and "
                f"'parameters' ({self._n_parameters}) does not match"
                )

        self._n_interactions = n_interactions

    def get_interaction(self, AINDEX index):
        """Return info for interaction

        Args:
            index: Index of the interaction to get the info for
            interaction: Interaction type

        Returns:
            Dictionary with keys according to
            :obj:`self._index_names` and :obj:`self._param_names` and
            corresponding values
        """

        if (index < 0) or (index >= (self._n_indices // self._interaction._dindex)):
            raise IndexError(
                "Interaction index out of range"
                )

        cdef dict info = {}
        cdef AINDEX i
        cdef str name

        for i, name in enumerate(self._interaction._index_names):
            info[name] = self._indices[index * self._interaction._dindex + i]

        for i, name in enumerate(self._interaction._param_names):
            info[name] = self._parameters[index * self._interaction._dparam + i]

        return info

    cdef (AINDEX*, AVALUE*) _next_interaction(self) nogil:
        """Return next indices/parameters combination

        Note:
            This is not save! No check is performed if the array storing
            indices and parameters have reached their end.
            Use :func:`_reset_iteration` to start iteration from the beginning
            and call :func:`_next_interaction` at most :attr:`_n_interactions`
            times. Beyond this, the function will return nonsense.
        """
        self._it = self._it +  1
        return (
            &self._indices[self._it * self._interaction._dindex],
            &self._parameters[self._it * self._interaction._dparam]
            )

    cdef void _reset_iteration(self) nogil:
        self._it = -1


cdef class NeighboursProvider(InteractionProvider):
    def __cinit__(
            self,
            system,
            table,
            *,
            **kwargs):

        self._system = system
        self._table = table

    cpdef void _check_consistency(self) except *:
        """Raise error if interaction does not require pairs"""

        if self._interaction._dindex != 2:
            raise ValueError(
                f"interaction {self._interaction} does not work on pairs "
                f"but provider {self} does only provide pairs"
                )

    cdef (AINDEX*, AVALUE*) _next_interaction(self) nogil:
        """Return next indices/parameters combination

        Note:
            This is not save! No check is performed if the array storing
            indices and parameters have reached their end.
            Use :func:`_reset_iteration` to start iteration from the beginning
            and call :func:`_next_interaction` at most :attr:`_n_interactions`
            times. Beyond this, the function will return nonsense.
        """
        cdef AINDEX* indices = self._system.neighbours._next_interaction()
        cdef AVALUE* parameters = self._table._get_parameters(indices)

        return (indices, parameters)

    cdef void _reset_iteration(self) nogil:
        self._n_interactions = self._system.neighbours._get_n_interactions()
        self._system.neighbours._reset_iteration()


cdef class Table:
    def __init__(self):
        if type(self) is Table:
            raise RuntimeError(
                f"Cannot instantiate abstract class {type(self)}"
                )

    cdef AVALUE* _get_parameters(self, AINDEX* indices) nogil: ...


cdef class DummyTable(Table):

    def __cinit__(self):
        self.parameters[0] = 0.3401
        self.parameters[1] = 0.978638

    cdef AVALUE* _get_parameters(self, AINDEX* indices) nogil:
        return &self.parameters[0]

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
            provider=None,
            *,
            group: int = 0,
            _id: Optional[int] = None,
            index_names: Optional[list] = None,
            param_names: Optional[list] = None,
            **kwargs):

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

        if type(self) is Interaction:
            raise RuntimeError(
                f"Cannot instantiate abstract class {type(self)}"
                )

        if index_names is None:
            index_names = self._default_index_names
        self._index_names = index_names

        if param_names is None:
            param_names = self._default_param_names
        self._param_names = param_names

        self._dindex = len(self._index_names)
        self._dparam = len(self._param_names)

        self.provider = provider

        if _id is None:
            _id = self._default_id
        self._id = _id

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
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, value):
        if value is None:
            value = NoProvider()
        self._provider = value
        self._provider._interaction = self
        self._provider._check_consistency()

    @property
    def id(self):
       return self._id

    cdef inline void _add_all_forces(self, System system) nogil:
        cdef AINDEX index
        cdef AINDEX *indices
        cdef AVALUE *parameters

        self._provider._reset_iteration()
        for index in prange(self._provider._n_interactions):
            indices, parameters = self._provider._next_interaction()
            self._add_force(indices, parameters, system)

    def add_force(self, indices: list, parameters: list, System system):
        cdef AINDEX i

        cdef AINDEX* indices_ptr = _allocate_and_fill_aindex_array(
            len(indices), indices
            )

        cdef AVALUE* parameters_ptr = _allocate_and_fill_avalue_array(
            len(parameters), parameters
            )

        self._add_force(indices_ptr, parameters_ptr, system)

        free(indices_ptr)
        free(parameters_ptr)

    cdef void _add_force(
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

        energy = self._get_energy(indices_ptr, parameters_ptr, system)

        free(indices_ptr)
        free(parameters_ptr)

        return energy

    cdef AVALUE _get_energy(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil: ...

    cdef inline AVALUE _get_total_energy(
            self,  System system) nogil:

        cdef AINDEX index
        cdef AINDEX *indices
        cdef AVALUE *parameters
        cdef AVALUE energy = 0

        self._provider._reset_iteration()
        for index in prange(self._provider._n_interactions):
            indices, parameters = self._provider._next_interaction()
            energy += self._get_energy(indices, parameters, system)

        return energy


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

    cdef void _add_force(
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

    cdef AVALUE _get_energy(
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

    cdef void _add_force(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        cdef AINDEX i
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]

        for i in range(d):
            f1[i] = 0

    cdef AVALUE _get_energy(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        return 0


cdef class Stabilizer(Interaction):
    """Removes negligible forces acting on atoms"""

    _default_index_names = ["p1"]
    _default_param_names = ["tolerance"]
    _default_id = 11

    cdef void _add_force(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        cdef AINDEX i
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE tolerance = parameters[0]

        for i in range(d):
            if f1[i] < tolerance:
                f1[i] = 0

    cdef AVALUE _get_energy(
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

    cdef void _add_force(
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

    cdef AVALUE _get_energy(
            self,
            AINDEX *indices,
            AVALUE *parameters,
            System system) nogil:

        return 0

cdef class HarmonicRepulsion(Interaction):
    """Harmonic spring force approximating a chemical bond"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["r0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects two particles with parameters r0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force(
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

        if r < r0:
            f = k * (r - r0)
            for i in range(d):
                _f = f * rv[i] / r
                f1[i] += _f
                f2[i] -= _f

    cdef AVALUE _get_energy(
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

        if r > r0:
            return 0

        return 0.5 * k * cpow(r - r0, 2)

cdef class HarmonicBond(Interaction):
    """Harmonic spring force approximating a chemical bond"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["r0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects two particles with parameters r0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force(
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

        f = k * (r - r0)
        for i in range(d):
            _f = f * rv[i] / r
            f1[i] += _f
            f2[i] -= _f

    cdef AVALUE _get_energy(
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

    cdef void _add_force(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rab, rcb, f
        cdef AVALUE costheta
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE *f3 = &system._forces_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvcb = system._resources.rvb
        cdef AVALUE *rvabn = system._resources.rvan
        cdef AVALUE *rvcbn = system._resources.rvbn
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system.pbc._pbc_distance(rvab, c1, c2, d)
        rab = _norm2(rvab, d)
        _normalise(rvab, rvabn, rab, d)

        system.pbc._pbc_distance(rvcb, c3, c2, d)
        rcb = _norm2(rvcb, d)
        _normalise(rvcb, rvcbn, rcb, d)

        costheta = _cosangle(rvabn, rvcbn, d)
        _derangle(
            costheta,
            rvabn, rvcbn, rab, rcb,
            der1, der2, der3, d
            )

        f = k * (cacos(costheta) - theta0)
        for i in range(d):
            f1[i] += f * der1[i]
            f2[i] += f * der2[i]
            f3[i] += f * der3[i]

    cdef AVALUE _get_energy(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rab, rcb
        cdef AVALUE costheta
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvcb = system._resources.rvb
        cdef AVALUE *rvabn = system._resources.rvan
        cdef AVALUE *rvcbn = system._resources.rvbn
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system.pbc._pbc_distance(rvab, c1, c2, d)
        rab = _norm2(rvab, d)
        _normalise(rvab, rvabn, rab, d)

        system.pbc._pbc_distance(rvcb, c3, c2, d)
        rcb = _norm2(rvcb, d)
        _normalise(rvcb, rvcbn, rcb, d)

        costheta = _cosangle(rvabn, rvcbn, d)

        return 0.5 * k * cpow(cacos(costheta) - theta0, 2)


cdef class CosineHarmonicAngle(Interaction):
    """Harmonic cosine force approximating a valence angle"""

    _default_index_names = ["p1", "p2", "p3"]
    _default_param_names = ["costheta0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects three particles with parameters costheta0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rab, rcb, f
        cdef AVALUE costheta
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE *f3 = &system._forces_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvcb = system._resources.rvb
        cdef AVALUE *rvabn = system._resources.rvan
        cdef AVALUE *rvcbn = system._resources.rvbn
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system.pbc._pbc_distance(rvab, c1, c2, d)
        rab = _norm2(rvab, d)
        _normalise(rvab, rvabn, rab, d)

        system.pbc._pbc_distance(rvcb, c3, c2, d)
        rcb = _norm2(rvcb, d)
        _normalise(rvcb, rvcbn, rcb, d)

        costheta = _cosangle(rvabn, rvcbn, d)
        _derangle(
            costheta,
            rvabn, rvcbn, rab, rcb,
            der1, der2, der3, d
            )

        f = -k * (costheta - theta0) * csin(cacos(costheta))
        for i in range(d):
            f1[i] += f * der1[i]
            f2[i] += f * der2[i]
            f3[i] += f * der3[i]

    cdef AVALUE _get_energy(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rab, rcb
        cdef AVALUE costheta
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE theta0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvcb = system._resources.rvb
        cdef AVALUE *rvabn = system._resources.rvan
        cdef AVALUE *rvcbn = system._resources.rvbn
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3

        system.pbc._pbc_distance(rvab, c1, c2, d)
        rab = _norm2(rvab, d)
        _normalise(rvab, rvabn, rab, d)

        system.pbc._pbc_distance(rvcb, c3, c2, d)
        rcb = _norm2(rvcb, d)
        _normalise(rvcb, rvcbn, rcb, d)

        costheta = _cosangle(rvabn, rvcbn, d)

        return 0.5 * k * cpow(costheta - theta0, 2)


cdef class HarmonicTorsion(Interaction):
    """Harmonic force approximating a (improper) dihedral angle"""

    _default_index_names = ["p1", "p2", "p3", "p4"]
    _default_param_names = ["phi0", "k"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects three particles with parameters theta0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rbc, rbcsq, usq, vsq, f
        cdef AVALUE phi
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *c4 = &system._configuration_ptr[indices[3] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE *f3 = &system._forces_ptr[indices[2] * d]
        cdef AVALUE *f4 = &system._forces_ptr[indices[3] * d]
        cdef AVALUE phi0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvbc = system._resources.rvb
        cdef AVALUE *rvcd = system._resources.rvc
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3
        cdef AVALUE *der4 = system._resources.der4
        cdef AVALUE *u = system._resources.u
        cdef AVALUE *v = system._resources.v

        system.pbc._pbc_distance(rvab, c1, c2, d)
        system.pbc._pbc_distance(rvbc, c2, c3, d)
        system.pbc._pbc_distance(rvcd, c3, c4, d)

        rbcsq = _norm2sq(rvbc, d)
        rbc = csqrt(rbcsq)

        _cross3(rvab, rvbc, u)
        _cross3(rvbc, rvcd, v)
        usq = _norm2sq(u, d)
        vsq = _norm2sq(v, d)

        phi = _torsion(rvab, u, v, rbc, d)

        _dertorsion(
            rvab, rvbc, rvcd,
            u, v, rbc, rbcsq, usq, vsq,
            der1, der2, der3, der4, d
            )

        f = -k * (phi - phi0)
        for i in range(d):
            f1[i] += f * der1[i]
            f2[i] += f * der2[i]
            f3[i] += f * der3[i]
            f4[i] += f * der4[i]

    cdef AVALUE _get_energy(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rbc
        cdef AVALUE phi
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *c4 = &system._configuration_ptr[indices[3] * d]
        cdef AVALUE phi0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvbc = system._resources.rvb
        cdef AVALUE *rvcd = system._resources.rvc
        cdef AVALUE *u = system._resources.u
        cdef AVALUE *v = system._resources.v

        system.pbc._pbc_distance(rvab, c1, c2, d)
        system.pbc._pbc_distance(rvbc, c2, c3, d)
        system.pbc._pbc_distance(rvcd, c3, c4, d)

        rbc = _norm2(rvbc, d)

        _cross3(rvab, rvbc, u)
        _cross3(rvbc, rvcd, v)

        phi = _torsion(rvab, u, v, rbc, d)

        return 0.5 * k * cpow(phi - phi0, 2)


cdef class PeriodicTorsion(Interaction):
    """Periodic force approximating a (improper) dihedral angle"""

    _default_index_names = ["p1", "p2", "p3", "p4"]
    _default_param_names = ["phi0", "k", "n"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        """Connects three particles with parameters theta0 and k"""
        super().__init__(*args, **kwargs)

    cdef void _add_force(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rbc, rbcsq, usq, vsq, f
        cdef AVALUE phi
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *c4 = &system._configuration_ptr[indices[3] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE *f3 = &system._forces_ptr[indices[2] * d]
        cdef AVALUE *f4 = &system._forces_ptr[indices[3] * d]
        cdef AVALUE phi0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE n = parameters[2]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvbc = system._resources.rvb
        cdef AVALUE *rvcd = system._resources.rvc
        cdef AVALUE *der1 = system._resources.der1
        cdef AVALUE *der2 = system._resources.der2
        cdef AVALUE *der3 = system._resources.der3
        cdef AVALUE *der4 = system._resources.der4
        cdef AVALUE *u = system._resources.u
        cdef AVALUE *v = system._resources.v

        system.pbc._pbc_distance(rvab, c1, c2, d)
        system.pbc._pbc_distance(rvbc, c2, c3, d)
        system.pbc._pbc_distance(rvcd, c3, c4, d)

        rbcsq = _norm2sq(rvbc, d)
        rbc = csqrt(rbcsq)

        _cross3(rvab, rvbc, u)
        _cross3(rvbc, rvcd, v)
        usq = _norm2sq(u, d)
        vsq = _norm2sq(v, d)

        phi = _torsion(rvab, u, v, rbc, d)

        _dertorsion(
            rvab, rvbc, rvcd,
            u, v, rbc, rbcsq, usq, vsq,
            der1, der2, der3, der4, d
            )

        f = -k * n * csin(n * phi - phi0)
        for i in range(d):
            f1[i] += f * der1[i]
            f2[i] += f * der2[i]
            f3[i] += f * der3[i]
            f4[i] += f * der4[i]

    cdef AVALUE _get_energy(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rbc
        cdef AVALUE phi
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *c3 = &system._configuration_ptr[indices[2] * d]
        cdef AVALUE *c4 = &system._configuration_ptr[indices[3] * d]
        cdef AVALUE phi0 = parameters[0]
        cdef AVALUE k = parameters[1]
        cdef AVALUE n = parameters[2]
        cdef AVALUE *rvab = system._resources.rva
        cdef AVALUE *rvbc = system._resources.rvb
        cdef AVALUE *rvcd = system._resources.rvc
        cdef AVALUE *u = system._resources.u
        cdef AVALUE *v = system._resources.v

        system.pbc._pbc_distance(rvab, c1, c2, d)
        system.pbc._pbc_distance(rvbc, c2, c3, d)
        system.pbc._pbc_distance(rvcd, c3, c4, d)

        rbc = _norm2(rvbc, d)

        _cross3(rvab, rvbc, u)
        _cross3(rvbc, rvcd, v)

        phi = _torsion(rvab, u, v, rbc, d)

        return k * (1 + ccos(n * phi - phi0))


cdef class LJ(Interaction):
    """Non-bonded Lennard-Jones interaction"""

    _default_index_names = ["p1", "p2"]
    _default_param_names = ["sigma", "epsilon"]
    _default_id = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    cdef void _add_force(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rsq, f, _f, sr2, sr6
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE *f1 = &system._forces_ptr[indices[0] * d]
        cdef AVALUE *f2 = &system._forces_ptr[indices[1] * d]
        cdef AVALUE s = parameters[0]
        cdef AVALUE e = parameters[1]
        cdef AVALUE *rv = system._resources.rva

        # Ref: f = 24 * e / r * (2 * cpow(s / r, 12) - cpow(s / r, 6))

        system.pbc._pbc_distance(rv, c1, c2, d)
        rsq = _norm2sq(rv, d)

        sr2 = (s * s) / rsq
        sr6 = sr2 * sr2 * sr2
        f = 24 * e * sr6 * (2 * sr6 - 1) / rsq
        for i in range(d):
            _f = f * rv[i]
            f1[i] -= _f
            f2[i] += _f

    cdef AVALUE _get_energy(
        self,
        AINDEX *indices, AVALUE *parameters,
        System system) nogil:

        cdef AINDEX i
        cdef AVALUE rsq, sr2, sr6
        cdef AINDEX d = system._dim_per_atom
        cdef AVALUE *c1 = &system._configuration_ptr[indices[0] * d]
        cdef AVALUE *c2 = &system._configuration_ptr[indices[1] * d]
        cdef AVALUE s = parameters[0]
        cdef AVALUE e = parameters[1]
        cdef AVALUE *rv = system._resources.rva

        # Ref: 4 * e * (cpow(s / r, 12) - cpow(s / r, 6))

        system.pbc._pbc_distance(rv, c1, c2, d)
        rsq = _norm2sq(rv, d)

        sr2 = (s * s) / rsq
        sr6 = sr2 * sr2 * sr2
        return 4 * e * sr6 * (sr6 - 1)

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


cdef class Cutoff:
    pass