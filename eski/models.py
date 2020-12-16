from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Model:
    structure: np.ndarray
    velocities: Optional[np.ndarray]
    desc: str


argon = Model(
    structure=np.array(
                [[0, 0, 0]]
                ),
    velocities=None,
    desc="One lonely argon atom"
)

registered_systems = {
    "Argon": argon
}
