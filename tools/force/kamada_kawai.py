# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from itertools import combinations

import numpy as np
from numpy import typing as npt

from frame.die.die import Die
from frame.netlist.netlist import Netlist

from tools.force.gekko_common import Model, solve_and_extract_solution

AdjacencyMatrix = npt.NDArray[np.float64]


def netlist_to_matrix(netlist: Netlist) -> AdjacencyMatrix:
    """
    Convert a netlist to a graph in adjacency matrix format
    :param netlist: the netlist
    :return:
    - graph: graph in adjacency matrix format, where graph[i][j] contains the weight of edge (i, j),
    or infinity if there is no such edge
    """
    mod2idx = {mod: idx for idx, mod in enumerate(netlist.modules)}
    graph = np.full((netlist.num_modules, netlist.num_modules), np.inf)
    for hyperedge in netlist.edges:
        for m1, m2 in combinations(hyperedge.modules, 2):
            i, j = mod2idx[m1], mod2idx[m2]
            graph[i][j] = hyperedge.weight
            graph[j][i] = hyperedge.weight
    return graph


def get_all_shortest_path_lengths(graph: AdjacencyMatrix) -> AdjacencyMatrix:
    """
    Get all-pairs shortest paths using Floyd-Warshall algorithm
    :param graph: the graph in adjacency matrix format, where graph[i][j] contains the weight of edge (i, j),
    or infinity if there is no such edge
    :return:
     - dist_mat: distance matrix, where dist_mat[i][j] contains the distance of the shortest path between nodes i and j,
     or infinity if there is no path
    """
    graph_order = len(graph)
    dist_mat = graph
    for k in range(graph_order):
        for i in range(graph_order):
            for j in range(graph_order):
                dist_mat[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
    return dist_mat


def kamada_kawai_layout(die: Die, verbose: bool = False, visualize: str | None = None, max_iter: int = 150) -> Die:
    """
    Relocate the modules of the netlist using and adaptation of the algorithm proposed
    by Kamada & Kawai in "An algorithm for drawing general undirected graphs" (1989).
    :param die: the die, with the netlist containing the modules
    :param verbose: if True, the GEKKO optimization log is displayed (not supported if visualize is True)
    :param visualize: if True, produce a GIF showing the optimization process
    :param max_iter: maximum number of iterations for GEKKO
    :return:
    - die: the die with the netlist with the modules relocated
    """
    assert die.netlist is not None, "No netlist associated to the die"

    graph = netlist_to_matrix(die.netlist)
    dist_mat = get_all_shortest_path_lengths(graph)

    graph_diameter = np.where(np.isinf(dist_mat), -np.inf, dist_mat).max()  # max value not infinity
    desirable_edge_length = ((die.width + die.height) / 2) / graph_diameter  # heuristic
    spring_length = desirable_edge_length * dist_mat
    spring_strength = np.reciprocal(dist_mat**2)

    m = Model(die)
    g = m.gekko  # Shortcut (reference)

    modules = die.netlist.modules
    for i in range(die.netlist.num_modules):
        for j in range(i, die.netlist.num_modules):
            if i != j:
                if spring_strength[i][j] != 0.0:
                    # Original Kamada-Kawai objective function
                    g.Minimize(spring_strength[i][j] *
                               ((m.x[i] - m.x[j])**2 + (m.y[i] - m.y[j])**2 + spring_length[i][j]**2
                                - 2 * spring_length[i][j] * g.sqrt((m.x[i] - m.x[j])**2 + (m.y[i] - m.y[j])**2)))
                # Repel modules from each other, depending on their area
                g.Maximize(0.01 *
                           ((m.x[i] - m.x[j])**2 + (m.y[i] - m.y[j])**2) / (modules[i].area() * modules[j].area()))
        # Repel modules from the die boundaries
        g.Minimize(0.5 *
                   (1 / m.x[i]**2 + 1 / (m.x[i] - die.width)**2 + 1 / m.y[i]**2 + 1 / (m.y[i] - die.height)**2))

    die = solve_and_extract_solution(m, die, verbose, visualize, max_iter)

    return die
