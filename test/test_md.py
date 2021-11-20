import numpy as np
import pytest

from eski import md, atoms


class TestSystem:

    @pytest.mark.parametrize(
        "registered_system_key",
        ["Dummy", "Argon2D", "Argon_pair3D"]
        )
    def test_create_with_data(
            self, registered_system_key, registered_system, file_regression):
        if registered_system.atoms is not None:
            registered_system.atoms = [
                atoms.Atom(*args, **kwargs)
                for args, kwargs in registered_system.atoms
                ]
            n_atoms = len(registered_system.atoms)
            dim_per_atom = registered_system.configuration.shape[0] // n_atoms
        else:
            n_atoms = 0
            dim_per_atom = 0

        system = md.System(
            registered_system.configuration,
            dim_per_atom=dim_per_atom,
            atoms=registered_system.atoms,
            desc=registered_system.desc
        )
        assert isinstance(system.desc, str)
        assert isinstance(system.configuration, np.ndarray)
        assert isinstance(system.velocities, np.ndarray)
        assert isinstance(system.forces, np.ndarray)
        assert system.configuration.shape == system.velocities.shape
        assert system.configuration.shape == system.forces.shape

        file_regression.check(repr(system))

    def test_fail_allocation_bad_natoms(self):
        with pytest.raises(AssertionError):
            md.System(
                np.zeros((3), dtype=float),
                dim_per_atom=1,
                atoms=[atoms.Atom() for _ in range(10)]
            )
