"""
Initial model, with small example.

See README.md for details.
"""

import os

import numpy as np
from gekko import GEKKO

from frame.geometry.geometry import Point, Shape, Rectangle
from plots import plot_result


def optimize(alpha: float, module_areas: list[float], wire_costs: dict[tuple[int, int], float],
             n_rows: int = 8, n_cols: int = 8, cell_width: float = 1, cell_height: float = 1):
    """
    Calculates the optimal floorplan given the module areas and the wire costs by unit length between them.
    The optimal floorplan is the configuration that minimizes the total wire length, WL, and the total dispersion of
    the modules, D:  alpha WL + (1 - alpha) D

    :param alpha: A number between 0 and 1 to balance the trade-off between the total wire length and dispersion
    :param module_areas: A list of the areas of each module
    :param wire_costs: The wire costs between each module. It should be a dictionary from tuple (edges, with the indexes
    in the module areas list as the ids) to float (the cost)
    :param n_rows: The number of rows in the cell grid
    :param n_cols: The number of columns in the cell grid
    :param cell_width: The width of the cells
    :param cell_height: The height of the cells
    :return:
        - ratios - An n_modules x (n_rows x n_cols) array containing the optimal ratio of each module
        in each cell in the cell grid.
        - centroids - An array of Point with the centroid of each module in the optimal floorplan.
        - dispersions - An array of floats with the dispersion of each module in the optimal floorplan.
        - wire_length - The total wire cost.
    """

    # Create (flattened) cell grid
    cells = [Rectangle()] * (n_rows * n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            cells[row * n_cols + col] = Rectangle(center=Point((col + 1 / 2) * cell_width, (row + 1 / 2) * cell_height),
                                                  shape=Shape(cell_width, cell_height))

    n_cells = n_rows * n_cols
    n_modules = len(module_areas)

    g = GEKKO(remote=False)

    # Centroid of modules
    x = g.Array(g.Var, n_modules, lb=0, ub=n_rows * cell_width)
    y = g.Array(g.Var, n_modules, lb=0, ub=n_cols * cell_width)

    # Dispersion of modules
    dx = g.Array(g.Var, n_modules, lb=0)
    dy = g.Array(g.Var, n_modules, lb=0)

    # Ratios of area of c used by module m
    a = g.Array(g.Var, (n_modules, n_cells), lb=0, ub=1)

    # Cell constraints
    for c in range(n_cells):
        # Cells cannot be over-occupied
        g.Equation(g.sum([a[m][c] for m in range(n_modules)]) <= 1)

    # Module constraints
    for m in range(n_modules):
        # Modules must have sufficient area
        g.Equation(g.sum([cells[c].area * a[m][c] for c in range(n_cells)]) >= module_areas[m])

        # Centroid of modules
        g.Equation(1 / module_areas[m] * g.sum([cells[c].area * cells[c].center.x * a[m][c]
                                               for c in range(n_cells)]) == x[m])
        g.Equation(1 / module_areas[m] * g.sum([cells[c].area * cells[c].center.y * a[m][c]
                                               for c in range(n_cells)]) == y[m])

        # Dispersion of modules
        g.Equation(g.sum([cells[c].area * a[m][c] * (x[m] - cells[c].center.x) ** 2
                          for c in range(n_cells)]) == dx[m])
        g.Equation(g.sum([cells[c].area * a[m][c] * (y[m] - cells[c].center.y) ** 2
                          for c in range(n_cells)]) == dy[m])

    # Objective function: alpha WL + (1 - alpha) D
    g.Minimize(
        alpha * g.sum([wire_costs[(m1, m2) if m1 < m2 else (m2, m1)] * ((x[m1] - x[m2]) ** 2 + (y[m1] - y[m2]) ** 2)
                       for m1 in range(n_modules) for m2 in range(n_modules) if m1 != m2]))
    g.Minimize((1 - alpha) * g.sum([dx[m] + dy[m] for m in range(n_modules)]))

    g.solve()

    # Extract solution
    ratios = np.empty((n_modules, n_cells))
    for m in range(n_modules):
        for c in range(n_cells):
            ratios[m][c] = a[m][c].value[0]
    ratios = np.reshape(ratios, (n_modules, n_rows, n_cols))

    centroids = np.empty(n_modules, dtype=Point)
    dispersions = np.empty(n_modules)
    for m in range(n_modules):
        centroids[m] = Point(x[m].value[0], y[m].value[0])
        dispersions[m] = dx[m].value[0] + dy[m].value[0]

    wire_length = sum([wire_costs[(m1, m2) if m1 < m2 else (m2, m1)] * (
            (centroids[m1] - centroids[m2]) & (centroids[m1] - centroids[m2]))
                       for m1 in range(n_modules) for m2 in range(n_modules) if m1 != m2])

    return ratios, centroids, dispersions, wire_length


def main():
    """Main function."""

    module_areas = [1, 2, 3, 12]
    wire_costs = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}
    n_cols, n_rows = 8, 8
    cell_width, cell_height = 1, 1

    results_dir = "results"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    for alpha in np.linspace(0, 1, 11):
        result = optimize(alpha, module_areas, wire_costs, n_rows, n_cols, cell_width, cell_height)
        ratios, centroids, dispersions, wire_length = result
        suptitle = f"alpha = {alpha:.1f}"
        print(suptitle)
        plot_result(module_areas, ratios, centroids, dispersions, wire_length,
                    n_rows, n_cols, cell_width, cell_height, suptitle, f"{results_dir}/fp-{alpha:.1f}.png")


if __name__ == "__main__":
    main()
