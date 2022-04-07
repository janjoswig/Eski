import numpy as np
import pytest

from eski import atoms, md, pbc


class TestSystem:

    @pytest.mark.parametrize(
        "registered_system_key",
        ["dummy", "argon2d", "argon_pair3d"]
        )
    def test_create_with_data(
            self, registered_system_key, registered_system, file_regression):

        assert isinstance(registered_system.desc, str)
        assert isinstance(registered_system.interactions, list)
        assert isinstance(registered_system.reporters, list)
        assert isinstance(registered_system._pbc, pbc.PBCHandler)

        s = registered_system.configuration.shape
        assert s == registered_system.velocities.shape
        assert s == registered_system.forces.shape

        assert isinstance(registered_system.dof, int)
        assert isinstance(registered_system.stop, int)
        assert isinstance(registered_system.step, int)
        assert isinstance(registered_system.target_step, int)
        assert isinstance(registered_system.total_mass, float)
        assert isinstance(registered_system.dim_per_atom, int)
        assert isinstance(registered_system.n_atoms, int)

        file_regression.check(repr(registered_system))

    def test_fail_allocation_bad_natoms(self):
        with pytest.raises(AssertionError):
            md.System(
                np.zeros((3), dtype=float),
                dim_per_atom=1,
                atoms=[atoms.Atom() for _ in range(10)]
            )
