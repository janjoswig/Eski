import numpy as np
import pytest

from eski import md, drivers


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
