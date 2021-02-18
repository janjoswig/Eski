import numpy as np
import pytest

from eski import drivers, atoms


class TestDriver:

    @pytest.mark.parametrize(
        "Driver,parameters",
        [
            (drivers.Driver, []),
            pytest.param(
                drivers.Driver, [0],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            (drivers.EulerIntegrator, [0.1]),
        ]
    )
    def test_create(self, Driver, parameters, file_regression):
        driver = Driver(parameters)
        file_regression.check(repr(driver))

    @pytest.mark.parametrize(
        "Driver,parameters",
        [
            (drivers.Driver, {}),
            (drivers.EulerIntegrator, {"dt": 0.1}),
        ]
    )
    def test_create_from_mapping(self, Driver, parameters):
        driver = Driver.from_mapping(parameters)
        assert isinstance(driver, Driver)

    @pytest.mark.parametrize(
        "Driver,parameters",
        [
            (drivers.EulerIntegrator, [0.1]),
        ]
    )
    def test_update(self, Driver, parameters, num_regression):
        driver = Driver(parameters)
        structure = np.array(
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
             )
        forcevectors = np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0],
             [0, 0, -1]], order="c", dtype=float
             )

        driver.update(
            structure,
            velocities,
            forcevectors,
            [atoms.Atom(mass=1) for _ in range(4)],
            4,
            )

        num_regression.check({
            "structure": structure.flatten(),
            "velocities": velocities.flatten(),
            })
