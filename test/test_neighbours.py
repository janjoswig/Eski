from eski import models, neighbours


def test_brute_force_verlet_list_on_argon_box():
    system = models.system_from_model("argon1000")
    system.neighbours = neighbours.NeighboursVerletBruteForce([0, 0.4])
    system.neighbours.update(system)

    assert system.neighbours.neighbourlist[0] == 6
    next_atom = system.neighbours.n_neighbours_positions[1]
    assert system.neighbours.neighbourlist[next_atom] == 5

    system.neighbours = neighbours.NeighboursVerletBruteForce([0, 0.35])
    system.neighbours.update(system)
    assert system.neighbours.neighbourlist[0] == 0
    assert system.neighbours.neighbourlist[1] == 0
