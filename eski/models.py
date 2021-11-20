from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np

from eski.atoms import Atom


@dataclass
class Model:
    configuration: np.ndarray
    velocities: Optional[np.ndarray]
    desc: Optional[str]
    atoms: Optional[Iterable]

    def make_Atoms(self):
        if self.atoms is None:
            return []

        return [Atom(*args, **kwargs) for args, kwargs in self.atoms]


dummy = Model(
    configuration=np.array([]),
    velocities=None,
    desc=None,
    atoms=None
)

argon2D = Model(
    configuration=np.array([0., 0.]),
    velocities=None,
    desc="One lonely argon atom",
    atoms=[(("Ar", ), {"mass": 40})]
)

argon_pair3D = Model(
    configuration=np.array([
        [0., 0., 0.],
        [1., 0., 0.]
        ]),
    velocities=None,
    desc="One lonely argon atom",
    atoms=[(("Ar", ), {"mass": 40}), (("Ar", ), {"mass": 40})]
)

registered_systems = {
    "Dummy": dummy,
    "Argon2D": argon2D,
    "Argon_pair3D": argon_pair3D
}
