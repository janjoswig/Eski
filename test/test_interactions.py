import numpy as np
import pytest

from eski import interactions


class TestInteractions:

    @pytest.mark.parametrize(
        "interaction_type,indices,parameters",
        [
            (interactions.Interaction, [1, 2, 3], [1.0, 1.0, 1.0]),
            pytest.param(
                interactions.Interaction, [1, 2, 3], [1.0],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                interactions.HarmonicBond, [1, 2, 3], [0.1, 0.5],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                interactions.HarmonicBond, [1, 2], [0.1, 0.5, 0.4],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                interactions.HarmonicBond, [1, 2], [0.1, 0.5, 0.4, 0.3],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            (interactions.HarmonicBond, [1, 2], [0.1, 0.5]),
            (interactions.LJ, [1, 2], [0.1, 0.5])
        ]
    )
    def test_create(self, interaction_type, indices, parameters, file_regression):
        interaction = interaction_type(indices, parameters)
        file_regression.check(repr(interaction))
        assert isinstance(interaction.id, int)

    @pytest.mark.parametrize(
        "interaction_type,mappings",
        [
            (interactions.Interaction, [{"p1": 0, "x": 0}, {"p1": 0, "x": 1}]),
            (
                interactions.HarmonicBond,
                [{"p1": 0, "p2": 1, "r0": 0.1, "k": 0.2}]
            ),
            pytest.param(
                interactions.HarmonicBond,
                [{"p1": 0, "p2": 1, "r0": 0.1}],
                marks=pytest.mark.raises(exception=KeyError)
                )
        ]
    )
    def test_create_from_mappings(self, interaction_type, mappings):
        interaction = interaction_type.from_mappings(mappings)
        assert isinstance(interaction, interaction_type)

    @pytest.mark.parametrize(
        "interaction_type,indices,parameters,i,expected",
        [
            (
                interactions.HarmonicBond, [1, 2], [0.1, 0.2], 0,
                {"p1": 1, "p2": 2, "r0": 0.1, "k": 0.2}
            ),
            pytest.param(
                interactions.HarmonicBond, [1, 2], [0.1, 0.2], 1,
                None, marks=pytest.mark.raises(exception=IndexError)
            ),
            pytest.param(
                interactions.Interaction, [0], [0], 0, {"p1": 0, "x": 0}
            )
        ]
    )
    def test_get_interaction(self, interaction_type, indices, parameters, i, expected):
        interaction = interaction_type(indices, parameters)
        assert expected == interaction.get_interaction(i)

    @pytest.mark.parametrize(
        "interaction_type,indices,parameters",
        [
            (interactions.HarmonicBond, [0, 1], [0.1, 0.1]),
            (
                interactions.HarmonicBond,
                [0, 1, 2, 3], [0.1, 0.1, 0.2, 0.1]
            ),
            (
                interactions.HarmonicBond,
                [0, 1, 0, 2, 2, 3],
                [0.1, 0.1, 0.2, 0.2, 0.2, 0.1]
            )
        ]
    )
    def test_add_contributions(
            self, interaction_type, indices, parameters, num_regression):
        interaction = interaction_type(indices, parameters)
        configuration = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
            ], order="c", dtype=float)

        forces = np.zeros_like(configuration, order="c", dtype=float)
        # force.add_contributions(configuration, forcevectors)

        num_regression.check({
            "interactions": forces.flatten()
            })
