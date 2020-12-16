import numpy as np
import pytest

from eski import md


class TestSystem:

    @pytest.mark.parametrize("registered_system_key", ["Argon"])
    def test_create_with_data(self, registered_system_key, registered_system):
        system = md.System(
            structure=registered_system.structure,
            desc=registered_system.desc
        )
        assert isinstance(system.desc, str)
        assert isinstance(system.structure, np.ndarray)
        assert isinstance(system.velocities, np.ndarray)
        assert system.structure.shape == system.velocities.shape