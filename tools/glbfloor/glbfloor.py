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
    parser = ArgumentParser(prog=prog, description="...")
    parser.add_argument("netlist",
                        help="input file (netlist)")
    parser.add_argument("-d", "--die", metavar="<width>x<height> or filename",
                        help="Size of the die (width x height) or name of the file")
    parser.add_argument("-g", "--grid", metavar="<rows>x<cols>", required=True,
                        help="Size of the initial grid (rows x columns)", )
    parser.add_argument("-a", "--alpha", type=float, required=True,
                        help="Hyperparameter for the tradeoff between dispersion and wire length")
    parser.add_argument("-p", "--plot",
                        help="Output plot file (image). If not present, no plot is produced")
    parser.add_argument("-o", "--outfile", required=True,
                        help="Output file (netlist)")
    return vars(parser.parse_args(args))


# TODO: think a better name, as the grid is still a netlist
def netlist_to_grid(netlist: Netlist, n_rows: int, n_cols: int, cell_shape: Shape, alpha: float) \
        -> tuple[Netlist, Allocation, dict[str, float]]:
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
    x = g.Array(g.Var, n_modules, lb=0, ub=n_rows * cell_shape.h)
    y = g.Array(g.Var, n_modules, lb=0, ub=n_cols * cell_shape.w)
    for m in range(netlist.num_modules):
        center = netlist.modules[m].center
        assert center is not None, "Modules should have initial center values. Maybe run the spectral tool?"
        x[m].value, y[m].value = center  # Initial values

    # Dispersion of modules
    dx = g.Array(g.Var, n_modules, lb=0)
    dy = g.Array(g.Var, n_modules, lb=0)

    # Ratios of area of c used by module m
    a = g.Array(g.Var, (n_modules, n_cells), lb=0, ub=1)

    # Cell constraints
    for c in range(n_cells):
        # Cells cannot be over-occupied
        g.Equation(g.sum([a[m][c] for m in range(n_modules)]) <= 1)

    mod2m = {}

    # Module constraints
    for m in range(netlist.num_modules):
        mod2m[netlist.modules[m]] = m

        m_area = netlist.modules[m].area()

        # Modules must have sufficient area
        g.Equation(g.sum([cells[c].area * a[m][c] for c in range(n_cells)]) >= m_area)

        # Centroid of modules
        g.Equation(1 / m_area * g.sum([cells[c].area * cells[c].center.x * a[m][c]
                                       for c in range(n_cells)]) == x[m])
        g.Equation(1 / m_area * g.sum([cells[c].area * cells[c].center.y * a[m][c]
                                       for c in range(n_cells)]) == y[m])

        # Dispersion of modules
        g.Equation(g.sum([cells[c].area * a[m][c] * (x[m] - cells[c].center.x)**2
                          for c in range(n_cells)]) == dx[m])
        g.Equation(g.sum([cells[c].area * a[m][c] * (y[m] - cells[c].center.y)**2
                          for c in range(n_cells)]) == dy[m])

    # Objective function: alpha * total wire length + (1 - alpha) * total dispersion

    # Total wire length
    for e in netlist.edges:
        if len(e.modules) == 2:
            m0 = mod2m[e.modules[0]]
            m1 = mod2m[e.modules[1]]
            g.Minimize(alpha * e.weight * ((x[m0] - x[m1])**2 + (y[m0] - y[m1])**2) / 2)
        else:
            ex = g.Var()
            g.Equation(g.sum([x[mod2m[mod]] for mod in e.modules]) / len(e.modules) == ex)
            ey = g.Var()
            g.Equation(g.sum([y[mod2m[mod]] for mod in e.modules]) / len(e.modules) == ey)
            for mod in e.modules:
                m = mod2m[mod]
                g.Minimize(alpha * e.weight * ((ex - x[m])**2 + (ey - y[m])**2))

    # Total dispersion
    g.Minimize((1 - alpha) * g.sum([dx[m] + dy[m] for m in range(n_modules)]))

    g.solve()

    # Extract solution
    allocation_list: list[None | list[list[float, float, float, float], dict[str, float]]] = [None] * n_cells
    for c in range(n_cells):
        c_alloc = Alloc()
        for m in range(n_modules):
            c_alloc[netlist.modules[m].name] = a[m][c].value[0]
        allocation_list[c] = [cells[c].vector_spec, c_alloc]
    allocation = Allocation(allocation_list)

    dispersions = {}
    for m in range(n_modules):
        module = netlist.modules[m]
        module.center = Point(x[m].value[0], y[m].value[0])
        dispersions[module.name] = dx[m].value[0] + dy[m].value[0]

    return netlist, allocation, dispersions


def refine_grid(netlist: Netlist, die_shape: Shape, n_rows: int, n_cols: int):
    raise NotImplemented  # TODO


def main(prog: str | None = None, args: list[str] | None = None):
    """Main function."""
    print("NOT YET IMPLEMENTED")

    options = parse_options(prog, args)

    # Die
    die_file = options["die"]
    if die_file is not None:
        die = Die(die_file)
        die_shape = Shape(die.width, die.height)
    else:
        die_shape = Shape(1, 1)

    infile = options["netlist"]

    # Initial grid
    n_rows, n_cols = map(int, options["grid"].split("x"))
    assert n_rows > 0 and n_cols > 0, "The number of rows and columns of the grid must be positive"

    alpha = float(options["alpha"])
    assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

    in_netlist = Netlist(infile)

    cell_shape = Shape(die_shape.w / n_rows, die_shape.h / n_cols)
    out_netlist, allocation, dispersions = netlist_to_grid(in_netlist, n_rows, n_cols, cell_shape, alpha)

    plot_file = options["plot"]
    if plot_file is not None:
        plot_grid(out_netlist, allocation, dispersions,
                  filename=plot_file, suptitle=f"alpha = {alpha}")

    # TODO

    outfile = options["outfile"]
    out_netlist.dump_yaml_netlist(outfile)


if __name__ == "__main__":
    main()
