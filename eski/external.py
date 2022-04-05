try:
    import mdtraj
except ModuleNotFoundError:
    pass


def system_to_mdtraj(system, atom_list, bonds=None):
    if bonds is None:
        bonds = []

    top = mdtraj.Topology()
    top.add_chain()
    top.add_residue("UNK", top.chain(0))

    for a in atom_list:
        top.add_atom(a.aname, mdtraj.element.get_by_symbol(a.element), top.residue(0))

    for a, b in bonds:
        top.add_bond(top.atom(a), top.atom(b))

    traj = mdtraj.Trajectory(
        system.configuration.reshape(system.n_atoms, system.dim_per_atom),
        top
        )

    return traj
