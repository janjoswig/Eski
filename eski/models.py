from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np

from eski.atoms import Atom
from eski.md import System
from eski import pbc


@dataclass
class Model:
    configuration: np.ndarray
    velocities: Optional[np.ndarray] = None
    desc: Optional[str] = None
    atoms: Optional[Iterable] = None
    pbc: Optional[tuple] = None

    def make_Atoms(self):
        if self.atoms is None:
            return None

        return [Atom(*args, **kwargs) for args, kwargs in self.atoms]

    def make_PBC(self):
        if self.pbc is None:
            return None

        pbctype, args, kwargs = self.pbc
        return getattr(pbc, pbctype)(*args, **kwargs)


def dummy():
    return Model(
        configuration=np.array([[]]),
        )


def argon2d():
    return Model(
        configuration=np.array([[0., 0.]]),
        desc="One lonely argon atom in 2D",
        atoms=[(("Ar", ), {"mass": 40})]
        )


def argon_pair3d():
    return Model(
        configuration=np.array([
            [0., 0., 0.],
            [1., 0., 0.]
            ]),
        desc="Two argon atoms in 3D",
        atoms=[(("Ar", ), {"mass": 40}), (("Ar", ), {"mass": 40})]
        )


def argon1000():
    # Argonbox OPLS/AA force field
    sigma = 0.340100               # nm
    # epsilon = 0.978638         # kJ/mol
    r0 = np.power(2, 1/6) * sigma  # nm
    delta_r = np.ceil(r0 * 100) / 100

    nx, ny, nz = 10, 10, 10
    configuration = np.array([
        [i * delta_r + delta_r / 2 for i in indices]
        for indices in grid_indices_3d(nx, ny, nz)
    ], dtype=float)

    bounds = np.array([nx * delta_r, ny * delta_r, nz * delta_r], dtype=float)
    atom_list = [(("Ar", ), {"mass": 40}) for _ in range(configuration.shape[0])]

    return Model(
        configuration=configuration,
        velocities=None,
        desc="A box filled with argon at close to equilibrium distance",
        atoms=atom_list,
        pbc=("OrthorhombicPBC", (bounds,), {})
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
        desc="A water molecule",
        atoms=[(("O", ), {"mass": 16}), (("H", ), {"mass": 1}), (("H", ), {"mass": 1})]
        )


def cc1d():
    return Model(
        configuration=np.array([
            [0.00],
            [0.1525],
            ]),
        desc="Two carbon atoms at equilibrium bond length in 1D",
        atoms=[(("C", ), {"mass": 12}), (("C", ), {"mass": 12})]
        )


registered_systems = {
    "dummy": dummy,
    "argon2d": argon2d,
    "argon_pair3d": argon_pair3d,
    "argon1000": argon1000,
    "screwed_water": screwed_water,
    "water": water,
    "cc1d": cc1d,
}


def system_from_model(model):
    if isinstance(model, str):
        model = registered_systems[model.lower()]()

    system = System(
        model.configuration,
        velocities=model.velocities,
        desc=model.desc,
        atoms=model.make_Atoms(),
        pbc=model.make_PBC()
        )

    return system


def grid_indices_3d(nx, ny, nz):
    yield from (
        (x, y, z)
        for x in range(nx)
        for y in range(ny)
        for z in range(nz)
        )