from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np

from eski.atoms import Atom
from eski.md import System


@dataclass
class Model:
    configuration: np.ndarray
    velocities: Optional[np.ndarray]
    desc: Optional[str]
    atoms: Optional[Iterable]

    def make_Atoms(self):
        if self.atoms is None:
            return None

        return [Atom(*args, **kwargs) for args, kwargs in self.atoms]


def dummy():
    return Model(
        configuration=np.array([[]]),
        velocities=None,
        desc=None,
        atoms=None
        )


def argon2d():
    return Model(
        configuration=np.array([[0., 0.]]),
        velocities=None,
        desc="One lonely argon atom in 2D",
        atoms=[(("Ar", ), {"mass": 40})]
        )


def argon_pair3d():
    return Model(
        configuration=np.array([
            [0., 0., 0.],
            [1., 0., 0.]
            ]),
        velocities=None,
        desc="Two argon atoms in 3D",
        atoms=[(("Ar", ), {"mass": 40}), (("Ar", ), {"mass": 40})]
        )


def screwed_water():
    return Model(
        configuration=np.array([
            [0.00, 0.0, 0.],
            [0.12, 0.0, 0.],
            [0.00, 0.1, 0.]
            ]),
        velocities=None,
        desc="A screwed water molecule",
        atoms=[(("O", ), {"mass": 16}), (("H", ), {"mass": 1}), (("H", ), {"mass": 1})]
        )


def water():
    return Model(
        configuration=np.array([
            0.01386173,  0.00429393, 0.,
            0.10931224, -0.00275818, 0.,
            -0.00317397,  0.09846425, 0.
            ]),
        velocities=None,
        desc="A water molecule",
        atoms=[(("O", ), {"mass": 16}), (("H", ), {"mass": 1}), (("H", ), {"mass": 1})]
        )


def cc1d():
    return Model(
        configuration=np.array([
            [0.00],
            [0.1525],
            ]),
        velocities=None,
        desc="Two carbon atoms at equilibrium bond length in 1D",
        atoms=[(("C", ), {"mass": 12}), (("C", ), {"mass": 12})]
        )


registered_systems = {
    "dummy": dummy,
    "argon2d": argon2d,
    "argon_pair3d": argon_pair3d,
    "screwed_water": screwed_water,
    "water": water,
    "cc1d": cc1d
}


def system_from_model(model):
    if isinstance(model, str):
        model = registered_systems[model.lower()]()

    system = System(
        model.configuration,
        velocities=model.velocities,
        desc=model.desc,
        atoms=model.make_Atoms(),
        )

    return system
