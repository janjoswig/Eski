import numpy as np
import pytest

from eski import interactions, models


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

    def test_screen_harmonic_bond(
            self, num_regression):
        system = models.system_from_model("cc1d")

        r0_list = np.linspace(0.0825, 0.2225, 101)
        energies = []
        for r0 in r0_list:
            system.interactions = [
                interactions.HarmonicBond([0, 1], [r0, 259408])
                ]
            energies.append(system.potential_energy())

        num_regression.check({
            "energies": np.asarray(energies).flatten()
            })

    def test_add_all_forces_cc1d(self, num_regression):
        system = models.system_from_model("cc1d")
        system.interactions = [
            interactions.HarmonicBond([0, 1], [0.16, 259408])
            ]
        system.add_all_forces()
        num_regression.check({
            "forces": system.forces
            })

    def test_add_all_forces_screwed_water(self, num_regression):
        system = models.system_from_model("screwed_water")
        system.interactions = [
            interactions.HarmonicBond(
                [0, 1, 0, 2],  [0.09572, 462750.4, 0.09572, 462750.4]
                )
            ]
        system.add_all_forces()
        num_regression.check({
            "forces": system.forces
            })
