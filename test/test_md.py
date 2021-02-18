import numpy as np
import pytest

from eski import md, atoms


class TestSystem:

    @pytest.mark.parametrize(
        "registered_system_key",
        ["Dummy", "Argon"]
        )
    def test_create_with_data(
            self, registered_system_key, registered_system, file_regression):
        if registered_system.atoms is not None:
            registered_system.atoms = [
                atoms.Atom(*args, **kwargs)
                for args, kwargs in registered_system.atoms
                ]

        system = md.System(
            structure=registered_system.structure,
            atoms=registered_system.atoms,
            desc=registered_system.desc
        )
        assert isinstance(system.desc, str)
        assert isinstance(system.structure, np.ndarray)
        assert isinstance(system.velocities, np.ndarray)
        assert isinstance(system.forcevectors, np.ndarray)
        assert system.structure.shape == system.velocities.shape
        assert system.structure.shape == system.forcevectors.shape
        assert system.structure.shape[0] == system.n_atoms

        file_regression.check(repr(system))

    def test_fail_allocation_bad_natoms(self):
        with pytest.raises(AssertionError):
            md.System(
                structure=np.zeros((1, 3), dtype=np.float),
                atoms=[atoms.Atom() for _ in range(10)]
            )
