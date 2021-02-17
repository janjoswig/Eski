import numpy as np
import pytest

from eski import md


class TestSystem:

    @pytest.mark.parametrize(
        "registered_system_key",
        ["Dummy", "Argon"]
        )
    def test_create_with_data(
            self, registered_system_key, registered_system):
        if registered_system.atoms is not None:
            registered_system.atoms = [
                md.Atom(*args, **kwargs)
                for args, kwargs in registered_system.atoms
                ]

        system = md.System(
            structure=registered_system.structure,
            atoms=registered_system.atoms,
            desc=registered_system.desc
        )
        assert isinstance(system.desc, str)
        assert isinstance(system.structure, np.ndarray)
        assert isinstance(system.velocities, np.ndarray)
        assert isinstance(system.forcevectors, np.ndarray)
        assert system.structure.shape == system.velocities.shape
        assert system.structure.shape == system.forcevectors.shape
        assert system.structure.shape[0] == system.n_atoms

        assert isinstance(str(system), str)

    def test_fail_allocation_bad_natoms(self):
        with pytest.raises(AssertionError):
            md.System(
                structure=np.zeros((1, 3), dtype=np.float),
                atoms=[md.Atom() for _ in range(10)]
            )


class TestAtom:

    @pytest.mark.parametrize(
        "args,kwargs,expected",
        [
            (("C"), {}, ("C", "C", "C", "UNK", 0., 0.)),
            (("Ar", "AX"), {"mass": 40}, ("Ar", "AX", "Ar", "UNK", 40., 0.))
            ]
        )
    def test_create(self, args, kwargs, expected):
        a = md.Atom(*args, **kwargs)
        check = dict(
            zip(
                ["aname", "atype", "element", "residue", "mass", "charge"],
                expected
               )
            )
        for attr, value in check.items():
            assert getattr(a, attr) == value

        attr_strs = [f"{k}={v}" for k, v in check.items()]
        assert repr(a) == f"{a.__class__.__name__}({', '.join(attr_strs)})"


class TestForce:

    @pytest.mark.parametrize(
        "Force,indices,parameters",
        [
            (md.Force, [1, 2, 3], []),
            pytest.param(
                md.Force, [1, 2, 3], [1.0],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                md.ForceHarmonicBond, [1, 2, 3], [0.1, 0.5],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                md.ForceHarmonicBond, [1, 2], [0.1, 0.5, 0.4],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            pytest.param(
                md.ForceHarmonicBond, [1, 2], [0.1, 0.5, 0.4, 0.3],
                marks=pytest.mark.raises(exception=ValueError)
                ),
            (md.ForceHarmonicBond, [1, 2], [0.1, 0.5])
        ]
    )
    def test_create(self, Force, indices, parameters):
        force = Force(indices, parameters)
        assert isinstance(str(force), str)
        assert isinstance(force.id, int)
        assert isinstance(force.n_interactions, int)

    @pytest.mark.parametrize(
        "Force,indices,parameters,i,expected",
        [
            (
                md.ForceHarmonicBond, [1, 2], [0.1, 0.2], 0,
                {"p1": 1, "p2": 2, "r0": 0.1, "k": 0.2}
            ),
            pytest.param(
                md.ForceHarmonicBond, [1, 2], [0.1, 0.2], 1,
                None, marks=pytest.mark.raises(exception=IndexError)
            ),
            pytest.param(
                md.Force, [0], [], 0, {"p1": 0}
            )
        ]
    )
    def test_get_interaction(self, Force, indices, i, parameters, expected):
        force = Force(indices, parameters)
        assert expected == force.get_interaction(i)

    @pytest.mark.parametrize(
        "Force,indices,parameters",
        [
            (md.ForceHarmonicBond, [0, 1], [0.1, 0.1]),
            (md.ForceHarmonicBond, [0, 1, 2, 3], [0.1, 0.1, 0.2, 0.1]),
            (
                md.ForceHarmonicBond,
                [0, 1, 0, 2, 2, 3],
                [0.1, 0.1, 0.2, 0.2, 0.2, 0.1]
            )
        ]
    )
    def test_add_contributions(
            self, Force, indices, parameters, num_regression):
        force = Force(indices, parameters)
        system = md.System([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

        force.add_contributions(system)

        num_regression.check({
            "forces": system.forcevectors.flatten()
            })


@pytest.mark.parametrize(
    "p1,p2,distance",
    [
        ([0, 0, 0], [0, 0, 0], 0),
        ([1, 0, 0], [0, 0, 0], 1),
        ([0, 0.5, 0], [0, -0.25, 0], 0.75)
    ]
)
def test_euclidean_distance(p1, p2, distance):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    assert distance == md.euclidean_distance(p1, p2)
