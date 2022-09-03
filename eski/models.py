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
            [0.01386173,  0.00429393, 0.],
            [0.10931224, -0.00275818, 0.],
            [-0.00317397,  0.09846425, 0.]
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


def arar1d():
    return Model(
        configuration=np.array([
            [0.00],
            [0.38174934263001775],
            ]),
        desc="Two argon atoms at equilibrium LJ distance in 1D",
        atoms=[(("Ar", ), {"mass": 40}), (("Ar", ), {"mass": 40})]
        )


def ethane():
    return Model(
        configuration=np.array([
            [-0.764326, 0.191356, -0.014663],
            [-0.612410, 0.191481, -0.014738],
            [-0.574807, 0.194137, -0.119181],
            [-0.574653, 0.099765,  0.035183],
            [-0.574800, 0.280635,  0.039729],
            [-0.801936, 0.102202, -0.069130],
            [-0.802083, 0.283072, -0.064584],
            [-0.801929, 0.188700,  0.089781],
        ]),
        desc="UFF optimised ethane molecule",
        atoms=[
            (("C", ), {"mass": 12}), (("C", ), {"mass": 12}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            ]
        )


def butane_gauche():
    return Model(
        configuration=np.array([
            [-0.765631,  0.077897, -0.007524],
            [-0.613193,  0.089512, -0.006731],
            [-0.561894,  0.213096,  0.068871],
            [-0.603440,  0.345263,  0.004227],
            [-0.804498,  0.074309,  0.096463],
            [-0.795106, -0.015617, -0.059640],
            [-0.812147,  0.162941, -0.061377],
            [-0.572286, -0.000907,  0.043350],
            [-0.575556,  0.091095, -0.111322],
            [-0.597703,  0.210378,  0.174079],
            [-0.450812,  0.209060,  0.070062],
            [-0.713634,  0.357608,  0.005216],
            [-0.566651,  0.350142, -0.100461],
            [-0.559219,  0.429613,  0.061318],
        ]),
        desc="UFF optimised butane molecule (gauche)",
        atoms=[
            (("C", ), {"mass": 12}), (("C", ), {"mass": 12}),
            (("C", ), {"mass": 12}), (("C", ), {"mass": 12}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            ]
        )


def butane_anti():
    return Model(
        configuration=np.array([
            [-0.679342,  0.146423,  0.000893],
            [-0.526759,  0.145156,  0.000843],
            [-0.472673,  0.001861, -0.004839],
            [-0.320091,  0.000593, -0.004889],
            [-0.718381,  0.091707,  0.089341],
            [-0.718381,  0.098882, -0.091611],
            [-0.715701,  0.251259,  0.005049],
            [-0.490432,  0.195183,  0.093266],
            [-0.490432,  0.202344, -0.087330],
            [-0.509001, -0.055327,  0.083333],
            [-0.509000, -0.048166, -0.097263],
            [-0.281051,  0.048135,  0.087614],
            [-0.281051,  0.055310, -0.093338],
            [-0.283731, -0.104242, -0.009046],
        ]),
        desc="UFF optimised butane molecule (anti)",
        atoms=[
            (("C", ), {"mass": 12}), (("C", ), {"mass": 12}),
            (("C", ), {"mass": 12}), (("C", ), {"mass": 12}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            (("H", ), {"mass": 1}), (("H", ), {"mass": 1}),
            ]
        )


registered_systems = {
    "dummy": dummy,
    "arar1d": arar1d,
    "argon2d": argon2d,
    "argon_pair3d": argon_pair3d,
    "argon1000": argon1000,
    "screwed_water": screwed_water,
    "water": water,
    "cc1d": cc1d,
    "ethane": ethane,
    "butane_gauche": butane_gauche,
    "butane_anti": butane_anti,
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