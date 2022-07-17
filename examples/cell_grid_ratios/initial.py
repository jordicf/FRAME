"""
Initial model, with small example.

See README.md for details.
"""

from __future__ import annotations

import os

import numpy as np
from gekko import GEKKO

from frame.geometry.geometry import Point, Shape, Rectangle
from plots import plot_result


def optimize(alpha: float, block_areas: list[float], wire_costs: dict[tuple[int, int], float],
             n_rows: int = 8, n_cols: int = 8, cell_width: float = 1, cell_height: float = 1):
    """
    Calculates the optimal floorplan given the block areas and the wire costs by unit length between them.
    The optimal floorplan is the configuration that minimizes the total wire length, WL, and the total dispersion of
    the blocks, D:  alpha WL + (1 - alpha) D.

    :param alpha: A number between 0 and 1 to balance the trade-off between the total wire length and dispersion.
    :param block_areas: A list of the areas of each block.
    :param wire_costs: The wire costs between each block. It should be a dictionary from tuple (edges, with the indexes
    in the block areas list as the ids) to float (the cost).
    :param n_rows: The number of rows in the cell grid.
    :param n_cols: The number of columns in the cell grid.
    :param cell_width: The width of the cells.
    :param cell_height: The height of the cells.
    :return:
        - ratios - An n_blocks x (n_rows x n_cols) array containing the optimal ratio of each block in each cell in the
        cell grid.
        - centroids - An array of Point with the centroid of each block in the optimal floorplan.
        - dispersions - An array of floats with the dispersion of each block in the optimal floorplan.
        - wire_length - The total wire cost.
    """

    # Create (flattened) cell grid
    cells = [Rectangle()] * (n_rows * n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            cells[row * n_cols + col] = Rectangle(center=Point((col + 1 / 2) * cell_width, (row + 1 / 2) * cell_height),
                                                  shape=Shape(cell_width, cell_height))

    n_cells = n_rows * n_cols
    n_blocks = len(block_areas)

    g = GEKKO(remote=False)

    # Centroid of blocks
    x = g.Array(g.Var, n_blocks, lb=0, ub=n_rows * cell_width)
    y = g.Array(g.Var, n_blocks, lb=0, ub=n_cols * cell_width)

    # Dispersion of blocks
    dx = g.Array(g.Var, n_blocks, lb=0)
    dy = g.Array(g.Var, n_blocks, lb=0)

    # Ratios of area of c used by block b
    a = g.Array(g.Var, (n_blocks, n_cells), lb=0, ub=1)

    # Cell constraints
    for c in range(n_cells):
        # Cells cannot be over-occupied
        g.Equation(g.sum([a[b][c] for b in range(n_blocks)]) <= 1)

    # Block constraints
    for b in range(n_blocks):
        # Blocks must have sufficient area
        g.Equation(g.sum([cells[c].area * a[b][c] for c in range(n_cells)]) >= block_areas[b])

        # Centroid of blocks
        g.Equation(1 / block_areas[b] * g.sum([cells[c].area * cells[c].center.x * a[b][c]
                                               for c in range(n_cells)]) == x[b])
        g.Equation(1 / block_areas[b] * g.sum([cells[c].area * cells[c].center.y * a[b][c]
                                               for c in range(n_cells)]) == y[b])

        # Dispersion of blocks
        g.Equation(g.sum([cells[c].area * a[b][c] * (x[b] - cells[c].center.x) ** 2
                          for c in range(n_cells)]) == dx[b])
        g.Equation(g.sum([cells[c].area * a[b][c] * (y[b] - cells[c].center.y) ** 2
                          for c in range(n_cells)]) == dy[b])

    # Objective function: alpha WL + (1 - alpha) D
    g.Minimize(
        alpha * g.sum([wire_costs[(b1, b2) if b1 < b2 else (b2, b1)] * ((x[b1] - x[b2]) ** 2 + (y[b1] - y[b2]) ** 2)
                       for b1 in range(n_blocks) for b2 in range(n_blocks) if b1 != b2]))
    g.Minimize((1 - alpha) * g.sum([dx[b] + dy[b] for b in range(n_blocks)]))

    g.solve()

    # Extract solution
    ratios = np.empty((n_blocks, n_cells))
    for b in range(n_blocks):
        for c in range(n_cells):
            ratios[b][c] = a[b][c].value[0]
    ratios = np.reshape(ratios, (n_blocks, n_rows, n_cols))

    centroids = np.empty(n_blocks, dtype=Point)
    dispersions = np.empty(n_blocks)
    for b in range(n_blocks):
        centroids[b] = Point(x[b].value[0], y[b].value[0])
        dispersions[b] = dx[b].value[0] + dy[b].value[0]

    wire_length = sum([wire_costs[(b1, b2) if b1 < b2 else (b2, b1)] * (
            (centroids[b1] - centroids[b2]) & (centroids[b1] - centroids[b2]))
                       for b1 in range(n_blocks) for b2 in range(n_blocks) if b1 != b2])

    return ratios, centroids, dispersions, wire_length


def main():
    """Main function."""

    block_areas = [1, 2, 3, 12]
    wire_costs = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}
    n_cols, n_rows = 8, 8
    cell_width, cell_height = 1, 1

    results_dir = "results"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    for alpha in np.linspace(0, 1, 11):
        result = optimize(alpha, block_areas, wire_costs, n_rows, n_cols, cell_width, cell_height)
        ratios, centroids, dispersions, wire_length = result
        suptitle = f"alpha = {alpha:.1f}"
        print(suptitle)
        plot_result(block_areas, wire_costs, ratios, centroids, dispersions, wire_length,
                    n_rows, n_cols, cell_width, cell_height, suptitle, f"{results_dir}/fp-{alpha:.1f}.png")


if __name__ == "__main__":
    main()
