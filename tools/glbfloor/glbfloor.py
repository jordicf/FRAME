from typing import Any
from argparse import ArgumentParser

from gekko import GEKKO

from frame.allocation.allocation import Allocation, Alloc
from frame.die.die import Die
from frame.geometry.geometry import Shape, Rectangle, Point
from frame.netlist.netlist import Netlist

from tools.glbfloor.plots import plot_grid


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="...")  # TODO: write description
    parser.add_argument("netlist",
                        help="input file (netlist)")
    parser.add_argument("-d", "--die", metavar="<width>x<height> or filename", default="1x1",
                        help="Size of the die (width x height) or name of the file")
    parser.add_argument("-g", "--grid", metavar="<rows>x<cols>", required=True,
                        help="Size of the initial grid (rows x columns)", )
    parser.add_argument("-a", "--alpha", type=float, required=True,
                        help="Tradeoff hyperparameter between 0 and 1 to control the balance between dispersion and "
                             "wire length")
    parser.add_argument("-t", "--threshold", type=float, default=0.95,
                        help="Threshold hyperparameter between 0 and 1 to decide if allocations must be refined")
    parser.add_argument("-p", "--plot",
                        help="Plot name. If not present, no plots are produced")
    parser.add_argument("--out-netlist",
                        help="Output netlist file. If not present, no file is produced")
    parser.add_argument("--out-allocation",
                        help="Output allocation file. If not present, no file is produced")
    return vars(parser.parse_args(args))


def add_objective(g, netlist: Netlist, alpha: float):
    # Objective function: alpha * total wire length + (1 - alpha) * total dispersion

    module2m = {}
    for m, module in enumerate(netlist.modules):
        module2m[module] = m

    # Total wire length
    for e in netlist.edges:
        if len(e.modules) == 2:
            m0 = module2m[e.modules[0]]
            m1 = module2m[e.modules[1]]
            g.Minimize(alpha * e.weight * ((g.x[m0] - g.x[m1])**2 + (g.y[m0] - g.y[m1])**2) / 2)
        else:
            ex = g.Var()
            g.Equation(g.sum([g.x[module2m[module]] for module in e.modules]) / len(e.modules) == ex)
            ey = g.Var()
            g.Equation(g.sum([g.y[module2m[module]] for module in e.modules]) / len(e.modules) == ey)
            for module in e.modules:
                m = module2m[module]
                g.Minimize(alpha * e.weight * ((ex - g.x[m])**2 + (ey - g.y[m])**2))

    # Total dispersion
    g.Minimize((1 - alpha) * g.sum([g.dx[m] + g.dy[m] for m in range(netlist.num_modules)]))

    return g


def calculate_initial_allocation(netlist: Netlist, n_rows: int, n_cols: int, cell_shape: Shape, alpha: float,
                                 plot_name: str | None = None) -> tuple[Netlist, Allocation]:
    """
    Calculates the initial legal floorplan allocation given the netlist, grid info, and the alpha hyperparameter
    :param netlist: netlist containing the modules with centroids initialized
    :param n_rows: initial number of rows in the grid
    :param n_cols: initial number of columns in the grid
    :param cell_shape: shape of the cells (typically the shape of the die scaled by the number of rows and columns)
    :param alpha: hyperparameter between 0 and 1 to control the balance between dispersion and wire length.
    Smaller values will reduce the dispersion and increase the wire length, and greater ones the other way around
    :param plot_name: name of the plot to be produced in each iteration The iteration number (0) and the PNG extension
    are added automatically. If None, no plots are produced.
    :return: the optimal solution found:
    - netlist - Netlist with the centroids of the modules updated.
    - allocation - Allocation with the ratio of each module in each cell of the grid.
    """
    cells = [Rectangle()] * (n_rows * n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            cells[row * n_cols + col] = Rectangle(
                center=Point((col + 1 / 2) * cell_shape.w, (row + 1 / 2) * cell_shape.h),
                shape=cell_shape)

    n_cells = n_rows * n_cols
    n_modules = netlist.num_modules

    g = GEKKO(remote=False)

    # Centroid of modules
    g.x = g.Array(g.Var, n_modules, lb=0, ub=n_rows * cell_shape.h)
    g.y = g.Array(g.Var, n_modules, lb=0, ub=n_cols * cell_shape.w)
    for m, module in enumerate(netlist.modules):
        center = module.center
        assert center is not None, "Modules should have initial center values. Maybe run the spectral tool?"
        g.x[m].value, g.y[m].value = center  # Initial values

    # Dispersion of modules
    g.dx = g.Array(g.Var, n_modules, lb=0)
    g.dy = g.Array(g.Var, n_modules, lb=0)

    # Ratios of area of c used by module m
    g.a = g.Array(g.Var, (n_modules, n_cells), lb=0, ub=1)

    # Cell constraints
    for c in range(n_cells):
        # Cells cannot be over-occupied
        g.Equation(g.sum([g.a[m][c] for m in range(n_modules)]) <= 1)

    # Module constraints
    for m in range(n_modules):
        m_area = netlist.modules[m].area()

        # Modules must have sufficient area
        g.Equation(g.sum([cells[c].area * g.a[m][c] for c in range(n_cells)]) >= m_area)

        # Centroid of modules
        g.Equation(1 / m_area * g.sum([cells[c].area * cells[c].center.x * g.a[m][c]
                                       for c in range(n_cells)]) == g.x[m])
        g.Equation(1 / m_area * g.sum([cells[c].area * cells[c].center.y * g.a[m][c]
                                       for c in range(n_cells)]) == g.y[m])

        # Dispersion of modules
        g.Equation(g.sum([cells[c].area * g.a[m][c] * (g.x[m] - cells[c].center.x)**2
                          for c in range(n_cells)]) == g.dx[m])
        g.Equation(g.sum([cells[c].area * g.a[m][c] * (g.y[m] - cells[c].center.y)**2
                          for c in range(n_cells)]) == g.dy[m])

    g = add_objective(g, netlist, alpha)
    g.solve()

    # Extract solution
    allocation_list: list[None | tuple[tuple[float, float, float, float], dict[str, float]]] = [None] * n_cells
    for c in range(n_cells):
        c_alloc = Alloc()
        for m, module in enumerate(netlist.modules):
            c_alloc[module.name] = g.a[m][c].value[0]
        allocation_list[c] = (cells[c].vector_spec, c_alloc)
    allocation = Allocation(allocation_list)

    dispersions = {}
    for m, module in enumerate(netlist.modules):
        module.center = Point(g.x[m].value[0], g.y[m].value[0])
        dispersions[module.name] = g.dx[m].value[0] + g.dy[m].value[0]

    if plot_name is not None:
        plot_grid(netlist, allocation, dispersions,
                  filename=f"{plot_name}-0.png", suptitle=f"alpha = {alpha}")

    return netlist, allocation


def optimize_allocation(netlist: Netlist, allocation: Allocation, threshold: float, alpha: float) \
        -> tuple[Netlist, Allocation, dict[str, float]]:
    raise NotImplementedError  # TODO


def refine_and_optimize_allocation(netlist: Netlist, allocation: Allocation, threshold: float, alpha: float,
                                   plot_name: str | None = None) -> tuple[Netlist, Allocation]:
    """
    Refine the given allocation and optimize it to minimize the dispersion and the wire length of the floor plan.
    The netlist, and the threshold and alpha hyperparameters are also required.
    :param netlist: netlist containing the modules with centroids initialized
    :param allocation: allocation with the ratio of each module in each cell of the grid, which possibly must be refined
    :param threshold: hyperparameter between 0 and 1 to decide if allocations must be refined
    :param alpha: hyperparameter between 0 and 1 to control the balance between dispersion and wire length.
    Smaller values will reduce the dispersion and increase the wire length, and greater ones the other way around
    :param plot_name: name of the plot to be produced in each iteration The iteration number and the PNG extension
    are added automatically. If None, no plots are produced.
    :return: the optimal solution found:
    - netlist - Netlist with the centroids of the modules updated.
    - allocation - Refined allocation with the ratio of each module in each cell of the grid.
    """
    n_iter = 1
    while allocation.must_be_refined(threshold):
        allocation = allocation.refine(threshold)
        netlist, allocation, dispersions = optimize_allocation(netlist, allocation, threshold, alpha)

        if plot_name is not None:
            plot_grid(netlist, allocation, dispersions,
                      filename=f"{plot_name}-{n_iter}.png", suptitle=f"alpha = {alpha}")
        n_iter += 1

    return netlist, allocation


def main(prog: str | None = None, args: list[str] | None = None):
    """Main function."""
    print("NOT YET IMPLEMENTED")  # this message is temporary

    options = parse_options(prog, args)

    # Initial netlist
    netlist = Netlist(options["netlist"])

    # Die shape
    die = Die(options["die"])
    die_shape = Shape(die.width, die.height)

    # Initial grid
    n_rows, n_cols = map(int, options["grid"].split("x"))
    assert n_rows > 0 and n_cols > 0, "The number of rows and columns of the grid must be positive"
    cell_shape = Shape(die_shape.w / n_rows, die_shape.h / n_cols)

    alpha = options["alpha"]
    assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

    threshold = options["threshold"]
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    netlist, allocation = calculate_initial_allocation(netlist, n_rows, n_cols, cell_shape, alpha, options["plot"])
    # netlist, allocation = refine_and_optimize_allocation(netlist, allocation, threshold, alpha, options["plot"])

    out_netlist_file = options["out_netlist"]
    if out_netlist_file is not None:
        netlist.write_yaml(out_netlist_file)

    out_allocation_file = options["out_allocation"]
    if out_allocation_file is not None:
        allocation.write_yaml(out_allocation_file)


if __name__ == "__main__":
    main()
