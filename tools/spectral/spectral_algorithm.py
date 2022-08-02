"""
Module for spectral layout. The algorithm implemented in this module is based on the one for Spectral Drawing
proposed by Yehuda Koren in his paper 'Drawing Graphs by Eigenvectors: Theory and Practice'. The algorithm has been
modified to incorporate the mass of each node. The mass is interpreted as the multiplicity of the node.
"""

import math
import random
from frame.utils.utils import Vector, Matrix
from .spectral_types import AdjEdge, AdjList


def spectral_layout_unit_square(adj: AdjList, mass: Vector, dim: int = 2) -> Matrix:
    """
    Computes a spectral layout of a graph in the square [+/-1,+/-1]
    :param adj: adjacency list (a list of edges for every node). Nodes are numbered from 0 to len(adj)-1
    :param mass: mass (size) of each node
    :param dim: number of dimensions of the layout
    :return: a matrix dim x nodes. Each vector represents the coordinates of one of the dimensions for each node
    """
    print("Adj =", adj)
    print("Mass =", mass)
    dim += 1
    epsilon = 1e-3
    one_minus_epsilon: float = 1 - epsilon  # tolerance for convergence
    n = len(adj)  # number of nodes
    degree: Vector = [sum([e.weight for e in adj[i]]) for i in range(n)]  # degree of each node (sum of edge weights)
    # Initial random coordinates for the nodes
    coord: Matrix = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(dim)]
    sum_mass = sum(mass)
    mass_factor = n / sum_mass
    scaled_mass: Vector = [mass_factor * x for x in mass]
    coord[0] = [1 / math.sqrt(n)] * n

    for k in range(1, dim):
        normalize(coord[k], scaled_mass)
        dotprod = 0.0
        num_iter = 0
        # Add a limit of iterations to reduce the CPU time
        while dotprod < one_minus_epsilon and num_iter < 100:
            # print("  Iter =", iter)
            num_iter += 1
            # print("  Normalized (K =", k, "):", coord[k])
            orthogonalize(coord, degree, scaled_mass, k)  # Orthogonalize coord[k] wrt to the other coordinates
            # print("  Orthogonalized (K =", k, "):", coord[k])
            new_coord = calculate_centroids(adj, coord[k], degree)
            # print("  Centroids (K =", k, "):", coord[k])
            # Sanity check (not all nodes in the same place). If so, a more modest move is done
            assert max(new_coord) - min(new_coord) > epsilon
            if max(new_coord) - min(new_coord) < epsilon:
                # new_coord = [random.uniform(-1, 1) for i in range(n)]
                new_coord = [0.5 * (new_coord[i] + coord[k][i]) for i in range(n)]
            normalize(new_coord, scaled_mass)
            dotprod = abs(dot_product(coord[k], new_coord, scaled_mass))
            # print("    Dotprod =", dotprod)
            coord[k] = new_coord
        print("Num iters =", num_iter)
        make_canonical(coord[k])
    print("Coord unit =", coord[1:dim])
    return coord[1:dim]


def normalize(x: Vector, mass: Vector) -> None:
    """
    Normalizes a vector. The normalization is done in-place
    :param x: the vector to be normalized
    :param mass: the mass of each node
    """
    norm = calculate_norm(x, mass)
    for i in range(len(x)):
        x[i] /= norm


def calculate_norm(x: Vector, mass: Vector) -> float:
    """
    Calculates the weighted norm of a vector
    :param x: the vector
    :param mass: the mass of each element
    :return: the norm
    """
    s = 0.0
    for i, v in enumerate(x):
        s += v * v * mass[i]
    return math.sqrt(s)


def orthogonalize(coord: Matrix, degree: Vector, mass: Vector, dim: int = 2) -> None:
    """
    Orthogonalizes coord[dim] with regard to coord[0..dim-1]. The orthogonalization is done in-place
    :param coord: coordinates of the nodes
    :param degree: degree of each node
    :param mass: mass of each node
    :param dim: dimension to be orthogonalized
    """
    n = len(degree)
    for k in range(dim):
        num, den = 0.0, 0.0
        for i in range(n):
            tmp = degree[i] * coord[k][i] * mass[i]
            num += coord[dim][i] * tmp
            den += coord[k][i] * tmp
        factor = num / den
        coord[dim] = [coord[dim][i] - factor * coord[k][i] for i in range(n)]


def calculate_centroids(adj: AdjList, coord: Vector, degree: Vector) -> Vector:
    """
    Calculates the new coordinates of the nodes
    :param adj: Adjacency list (with weights)
    :param coord: coordinates of the nodes
    :param degree: degree of each node
    :return: the new coordinates
    """
    n = len(coord)
    new_coord: Vector = [0] * n
    for i in range(n):
        center = sum([e.weight * coord[e.node] for e in adj[i]]) / degree[i]
        new_coord[i] = 0.5 * (coord[i] + center)
    return new_coord


def dot_product(v1: Vector, v2: Vector, weight: Vector) -> float:
    """
    Returns the weighted dot product of two vectors
    :param v1: first vector
    :param v2: second vector
    :param weight: weight of each element
    :return: the weighted dot product
    """
    return sum(weight[i] * v1[i] * v2[i] for i in range(len(v1)))


def make_canonical(coord: Vector) -> None:
    """
    Guarantees a canonical solution: the first coordinate is always positive.
    The canonicalization is done in place
    :param coord: list of coordinates
    """
    if coord[0] < 0:
        for i in range(len(coord)):
            coord[i] = -coord[i]


def check_orthogonal(coord: Matrix, mass: Vector) -> Vector:
    dim = len(coord)
    prod = []
    for d1 in range(1, dim):
        for d2 in range(d1):
            prod.append(dot_product(coord[d1], coord[d2], mass))
    return prod


def test_unbalancedsquare():
    """
    Test with four nodes connected as a square.
    """
    mass = [1.0] * 4
    mass[0] = 100.0
    adj = [[AdjEdge(1, 1), AdjEdge(3, 1)],
           [AdjEdge(0, 1), AdjEdge(2, 1)],
           [AdjEdge(1, 1), AdjEdge(3, 1)],
           [AdjEdge(0, 1), AdjEdge(2, 1)]]
    return adj, mass


def test_square():
    """
    Test with four nodes connected as a square.
    """
    mass = [1] * 4
    adj = [[AdjEdge(1, 1), AdjEdge(3, 1)],
           [AdjEdge(0, 1), AdjEdge(2, 1)],
           [AdjEdge(1, 1), AdjEdge(3, 1)],
           [AdjEdge(0, 1), AdjEdge(2, 1)]]
    return adj, mass


def test_hypersquare():
    """
    Test with four nodes connected as a square.
    """
    mass = [1.0] * 5
    # mass[4] = 0
    adj = [[AdjEdge(1, 1), AdjEdge(4, 4)],
           [AdjEdge(0, 1), AdjEdge(2, 1)],
           [AdjEdge(1, 1), AdjEdge(3, 1)],
           [AdjEdge(4, 4), AdjEdge(2, 1)],
           [AdjEdge(3, 4), AdjEdge(0, 4)]]
    return adj, mass


def test_triangle():
    mass = [1.0, 1.0, 1.0]
    adj = [[AdjEdge(1, 1), AdjEdge(2, 1)],
           [AdjEdge(0, 1), AdjEdge(2, 1)],
           [AdjEdge(0, 1), AdjEdge(1, 1)]
           ]
    return adj, mass


def test_chain(n):
    mass = [1.0] * n
    adj = [[AdjEdge(1, 1)]]
    for i in range(1, n - 1):
        adj.append([AdjEdge(i - 1, 1), AdjEdge(i + 1, 1)])
    adj.append([AdjEdge(n - 2, 1)])
    return adj, mass


def test_ring(n):
    mass = [1.0] * n
    adj = []
    for i in range(n):
        adj.append([AdjEdge((i - 1) % n, 1), AdjEdge((i + 1) % n, 1)])
    return adj, mass


def test_2nodes():
    mass = [2.0, 1.0]
    adj = [[AdjEdge(1, 1)], [AdjEdge(0, 1)]]
    return adj, mass
