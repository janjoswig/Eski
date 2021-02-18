import numpy as np
import pytest

from eski import forces


class TestForce:

    @pytest.mark.parametrize(
        "Force,indices,parameters",
        [
            (forces.Force, [1, 2, 3], []),
            pytest.param(
                forces.Force, [1, 2, 3], [1.0],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                forces.ForceHarmonicBond, [1, 2, 3], [0.1, 0.5],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                forces.ForceHarmonicBond, [1, 2], [0.1, 0.5, 0.4],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                forces.ForceHarmonicBond, [1, 2], [0.1, 0.5, 0.4, 0.3],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            (forces.ForceHarmonicBond, [1, 2], [0.1, 0.5])
        ]
    )
    def test_create(self, Force, indices, parameters, file_regression):
        force = Force(indices, parameters)
        file_regression.check(repr(force))
        assert isinstance(force.id, int)

    @pytest.mark.parametrize(
        "Force,forces",
        [
            (forces.Force, [{"p1": 0}, {"p1": 1}]),
            (forces.ForceHarmonicBond, [{"p1": 0, "p2": 1, "r0": 0.1, "k": 0.2}]),
            pytest.param(
                forces.ForceHarmonicBond,
                [{"p1": 0, "p2": 1, "r0": 0.1}],
                marks=pytest.mark.raises(exception=KeyError)
                )
        ]
    )
    def test_create_from_mappings(self, Force, forces):
        force = Force.from_mappings(forces)
        assert isinstance(force, Force)

    @pytest.mark.parametrize(
        "Force,indices,parameters,i,expected",
        [
            (
                forces.ForceHarmonicBond, [1, 2], [0.1, 0.2], 0,
                {"p1": 1, "p2": 2, "r0": 0.1, "k": 0.2}
            ),
            pytest.param(
                forces.ForceHarmonicBond, [1, 2], [0.1, 0.2], 1,
                None, marks=pytest.mark.raises(exception=IndexError)
            ),
            pytest.param(
                forces.Force, [0], [], 0, {"p1": 0}
            )
        ]
    )
    def test_get_interaction(self, Force, indices, i, parameters, expected):
        force = Force(indices, parameters)
        assert expected == force.get_interaction(i)

    @pytest.mark.parametrize(
        "Force,indices,parameters",
        [
            (forces.ForceHarmonicBond, [0, 1], [0.1, 0.1]),
            (forces.ForceHarmonicBond, [0, 1, 2, 3], [0.1, 0.1, 0.2, 0.1]),
            (
                forces.ForceHarmonicBond,
                [0, 1, 0, 2, 2, 3],
                [0.1, 0.1, 0.2, 0.2, 0.2, 0.1]
            )
        ]
    )
    def test_add_contributions(
            self, Force, indices, parameters, num_regression):
        force = Force(indices, parameters)
        structure = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]], order="c", dtype=float)

        forcevectors = np.zeros_like(structure, order="c", dtype=float)
        force.add_contributions(structure, forcevectors)

        num_regression.check({
            "forces": forcevectors.flatten()
            })
