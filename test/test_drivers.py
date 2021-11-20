import numpy as np
import pytest

from eski import drivers, atoms


class TestDriver:

    @pytest.mark.parametrize(
        "driver_type,parameters",
        [
            (drivers.Driver, []),
            pytest.param(
                drivers.Driver, [0],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            (drivers.EulerIntegrator, [0.1]),
        ]
    )
    def test_create(self, driver_type, parameters, file_regression):
        driver = driver_type(parameters)
        file_regression.check(repr(driver))

    @pytest.mark.parametrize(
        "driver_type,parameters",
        [
            (drivers.Driver, {}),
            (drivers.EulerIntegrator, {"dt": 0.1}),
        ]
    )
    def test_create_from_mapping(self, driver_type, parameters):
        driver = driver_type.from_mapping(parameters)
        assert isinstance(driver, driver_type)

    @pytest.mark.parametrize(
        "driver_type,parameters",
        [
            (drivers.EulerIntegrator, [0.1]),
        ]
    )
    def test_update(self, driver_type, parameters, num_regression):
        driver = driver_type(parameters)
        configuration = np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]], order="c", dtype=float
            )
        velocities = np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [1, 0, 0],
             [0, 0, 1]], order="c", dtype=float
            ).reshape(-1)
        forces = np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0],
             [0, 0, -1]], order="c", dtype=float
            ).reshape(-1)

        n_atoms = configuration.shape[0]
        dim_per_atom = configuration.shape[1]
        configuration = configuration.reshape(-1)
        n_dim = configuration.shape[0]

        support = {
            "n_atoms": n_atoms,
            "n_dim": n_dim,
            "dim_per_atom": dim_per_atom,
            }

        driver.update(
            configuration,
            velocities,
            forces,
            [atoms.Atom(mass=1) for _ in range(n_atoms)],
            support,
            )

        num_regression.check({
            "configuration": configuration.flatten(),
            "velocities": velocities.flatten(),
            })
