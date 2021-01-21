from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np


@dataclass
class Model:
    structure: np.ndarray
    velocities: Optional[np.ndarray]
    desc: Optional[str]
    atoms: Optional[Iterable]


dummy = Model(
    structure=np.array([], ndmin=2),
    velocities=None,
    desc=None,
    atoms=None
)

argon = Model(
    structure=np.array(
                [[0, 0, 0]]
                ),
    velocities=None,
    desc="One lonely argon atom",
    atoms=[(("Ar", ), {"mass": 40})]
)

registered_systems = {
    "Dummy": dummy,
    "Argon": argon
}
