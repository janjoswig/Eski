import numpy as np
import pytest

from eski import atoms, drivers, interactions, md
from eski.models import system_from_model


class TestDriver:

    @pytest.mark.parametrize(
        "driver_type,parameters",
        [
            pytest.param(
                drivers.Driver, [0],
                marks=pytest.mark.raises(exception=RuntimeError)
                ),
            (drivers.EulerIntegrator, [0.1]),
            (drivers.EulerMaruyamaIntegrator, [0.1, 0.001, 300]),
            (drivers.SteepestDescentMinimiser, [0.01, 100, 1.2, 0.2])
        ]
    )
    def test_create(self, driver_type, parameters, file_regression):
        driver = driver_type(parameters)
        file_regression.check(repr(driver))

    @pytest.mark.parametrize(
        "driver_type,parameters",
        [
            (drivers.EulerIntegrator, {"dt": 0.1}),
            (drivers.EulerMaruyamaIntegrator, {"dt": 0.1, "friction": 0.001, "T": 300}),
            (drivers.SteepestDescentMinimiser, {"tau": 0.01, "tolerance": 100, "tuneup": 1.2, "tunedown": 0.2})
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
             [0, 0, 2]], order="c", dtype=float
            ).reshape(-1)

        system = md.System(
            configuration,
            velocities=velocities,
            atoms=[atoms.Atom(mass=1) for _ in range(configuration.shape[0])]
            )

        driver.update(system)

        num_regression.check({
            "configuration": system.configuration,
            "velocities": system.velocities,
            })

    def test_steepest_descent_water(self):

        water = system_from_model("screwed_water")
        water.interactions = [
            interactions.HarmonicBond.from_explicit(
                [0, 1, 0, 2], [0.09572, 462750.4, 0.09572, 462750.4]
                ),
            interactions.HarmonicAngle.from_explicit(
                [1, 0, 2], [np.radians(104.520), 836.800]
                )
            ]
        water.drivers = [drivers.SteepestDescentMinimiser([0.01, 10, 1.2, 0.2])]

        water.simulate(100)
        assert water.potential_energy() < 0.001
        assert np.isclose(water.distance(0, 1), 0.09572, atol=1e-4)
        assert np.isclose(water.distance(0, 2), 0.09572, atol=1e-4)
