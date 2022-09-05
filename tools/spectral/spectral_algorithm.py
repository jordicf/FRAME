# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Module for spectral layout. The algorithm implemented in this module is based on the one for Spectral Drawing
proposed by Yehuda Koren in his paper 'Drawing Graphs by Eigenvectors: Theory and Practice'. The algorithm has been
modified to incorporate the mass of each node. The mass is interpreted as the multiplicity of the node.
"""

import math
import random
from frame.utils.utils import Vector, Matrix
from .spectral_types import AdjList


def spectral_layout_unit_square(adj: AdjList, mass: Vector, size: Vector, dim, fixed: Matrix) \
        -> tuple[Matrix, float, list[int]]:
    """
    Computes a spectral layout of a graph in the square [+/-1,+/-1]
    :param adj: adjacency list (a list of edges for every node). Nodes are numbered from 0 to len(adj)-1
    :param mass: mass (size) of each node
    :param size: length of each dimension (usually width and height)
    :param dim: number of dimensions of the layout
    :param fixed: a matrix with the coordinates of the fixed nodes. If the coordinates are negative, the nodes
                  are floating
    :return: a matrix dim x nodes. Each vector represents the coordinates of one of the dimensions for each node.
             It also returns the total wirelength and the number of iterations used for each dimension
    """
    # print("Adj =", adj)
    # print("Mass =", mass)
    dim += 1
    new_size = [2.0] + size
    n = len(adj)  # number of nodes
    epsilon = max(size) * n * 1e-10
    one_minus_epsilon: float = 1 - epsilon  # tolerance for convergence
    degree = [sum([e.weight for e in adj[i]]) for i in range(n)]  # degree of each node (sum of edge weights)
    radius = [math.sqrt(mass[i] / math.pi) for i in range(n)]  # radius of each node, assuming it is a circle
    max_span = [[new_size[d] / 2 - radius[i] for i in range(n)] for d in
                range(dim)]  # max span of each node in each dimension
    # Initial random coordinates for the nodes (always inside the die)
    coord: Matrix = [[random.uniform(-max_span[d][i], max_span[d][i]) for i in range(n)] for d in range(dim)]
    coord[0] = [1] * n

    # Fixed coordinates
    is_fixed: list[bool] = [x >= 0 for x in fixed[0]]
    # Apply the fixed coordinates (center the coordinates)
    for i in range(n):
        if is_fixed[i]:
            for d in range(1, dim):
                coord[d][i] = fixed[d - 1][i] - new_size[d] / 2

    float_mass = [0 if is_fixed[i] else mass[i] for i in range(n)]

    iterations = []
    for d in range(1, dim):
        # Apply the fixed coordinates
        normalize(coord[d], max_span[d], is_fixed)
        dotprod = 0.0
        num_iter = 0
        # Add a limit of iterations to reduce the CPU time
        while (dotprod < one_minus_epsilon or dotprod > 1 + epsilon) and num_iter < 10000:
            # print("  Iter =", d, num_iter)
            num_iter += 1
            # print("  Normalized (D =", d, "):", coord[d])
            orthogonalize(coord, float_mass, d, is_fixed)  # Orthogonalize coord[d] wrt to the other coordinates
            # print("  Orthogonalized (D =", d, "):", coord[k])
            tmp_coord = calculate_centroids(adj, coord[d], degree)
            # Keep the fixed coordinates
            new_coord = [coord[d][i] if is_fixed[i] else tmp_coord[i] for i in range(n)]
            # print("  Centroids (D =", d, "):", new_coord)
            # Sanity check (not all nodes in the same place). If so, a more modest move is done
            # print("diff =", max(new_coord) - min(new_coord))
            # assert max(new_coord) - min(new_coord) > epsilon
            if max(new_coord) - min(new_coord) < epsilon:
                # new_coord = [random.uniform(-1, 1) for i in range(n)]
                new_coord = [0.5 * (new_coord[i] + coord[d][i]) for i in range(n)]
            normalize(new_coord, max_span[d], is_fixed)
            dotprod = abs_norm_dot_product(coord[d], new_coord, float_mass)
            # print("    Dotprod =", dotprod)
            coord[d] = new_coord
        iterations.append(num_iter)
        # print("Num iters =", num_iter)
        # make_canonical(coord[k])
    # print("Coord unit =", coord[1:dim])
    final_coord = coord[1:dim]
    return final_coord, wirelength(adj, final_coord), iterations


def wirelength(adj: AdjList, coord: Matrix) -> float:
    """Returns the Manhattan wirelength"""
    ndim, nmod = len(coord), len(coord[0])
    total = 0.0
    for i in range(nmod):
        for e in adj[i]:
            total += e.weight * sum(abs(coord[d][e.node] - coord[d][i]) for d in range(ndim))
    return total / 2


def normalize(x: Vector, max_span: Vector, is_fixed: list[bool]) -> None:
    """
    Normalizes a vector by scaling the coordinates such that they fit inside the die.
    The normalization is done in-place. The fixed nodes are not modified
    :param x: the vector to be normalized
    :param max_span: the maximum span of each node
    :param is_fixed: indicates which nodes are fixed
    """
    scale = min(max_span[i] / abs(x[i]) for i in range(len(x)) if not is_fixed[i] and abs(x[i]) > 10e-10)
    for i in range(len(x)):
        if not is_fixed[i]:
            x[i] *= scale


def orthogonalize(coord: Matrix, mass: Vector, dim: int, is_fixed: list[bool]) -> None:
    """
    Orthogonalizes coord[dim] with regard to coord[0..dim-1]. The orthogonalization is done in-place
    :param coord: coordinates of the nodes
    :param mass: mass of each node
    :param dim: dimension to be orthogonalized
    :param is_fixed: indicates which nodes are fixed
    """
    n = len(mass)
    for k in range(dim):
        num, den = 0.0, 0.0
        for i in range(n):
            if not is_fixed[i]:
                # tmp = degree[i] * coord[k][i] * mass[i]
                tmp = coord[k][i] * mass[i]
                num += coord[dim][i] * tmp
                den += coord[k][i] * tmp
        factor = num / den
        coord[dim] = [coord[dim][i] if is_fixed[i] else coord[dim][i] - factor * coord[k][i] for i in range(n)]
        dotprod = abs_norm_dot_product(coord[dim], coord[k], mass)
        assert dotprod < 10e-12
        # print("dotprod =", dotprod)


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


def abs_norm_dot_product(v1: Vector, v2: Vector, weight: Vector) -> float:
    """
    Returns the normalized weighted dot product of two vectors
    :param v1: first vector
    :param v2: second vector
    :param weight: weight of each element
    :return: the normalized weighted dot product
    """
    dotprod = sum(weight[i] * v1[i] * v2[i] for i in range(len(v1)))
    norm = sum(weight[i] * v1[i] * v1[i] for i in range(len(v1)))
    return abs(dotprod / norm)


def check_orthogonal(coord: Matrix, mass: Vector) -> Vector:
    dim = len(coord)
    prod = []
    for d1 in range(1, dim):
        for d2 in range(d1):
            prod.append(abs_norm_dot_product(coord[d1], coord[d2], mass))
    return prod

