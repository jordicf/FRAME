# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""A netlist generator."""

import argparse
import random
from typing import Any
from ruamel.yaml import YAML

from frame.die.die import Die
from frame.geometry.geometry import Shape
from frame.utils.keywords import KW


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, description="A netlist generator.", usage='%(prog)s [options]')
    parser.add_argument("-o", "--outfile", help="output file (netlist)", required=True)
    parser.add_argument("--type", type=str,
                        choices=['grid', 'chain', 'ring', 'star', 'ring-star', 'one-net', 'htree'], required=True,
                        help="type of netlist (grid, chain, ring, star, ring-star, one-net, htree)")
    parser.add_argument("--size", type=int, nargs='+', required=True, help="size of the netlist")
    parser.add_argument("--add-centers", action="store_true",
                        help="add module centers (only supported for grid type, and requires to specify the die)")
    parser.add_argument("--add-noise", metavar="STANDARD DEVIATION", type=float, nargs="?", const=0.1, default=0,
                        help="(used only if --add-centers is present) adds random gaussian noise to the centers")
    parser.add_argument("--seed", type=int,
                        help="(used only if --add-noise is present) "
                             "integer number used as a seed for the random number generator")
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME",
                        help="(used only if --add-centers is present) "
                             "size of the die (width x height) or name of the file")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """Main function."""
    options = parse_options(prog, args)

    size = options['size']
    nsize = len(size)
    net_type = options['type']
    if net_type == 'grid':
        assert nsize == 2, "Two parameters must be specified for the size of a grid"
    else:
        assert nsize == 1, "Too many parameters for the size of the netlist"

    add_centers = options["add_centers"]
    die_shape = None
    sd = 0
    if add_centers:
        assert net_type == 'grid', "--add-center is only supported for the grid type"
        assert options['die'] is not None, "The die must be specified (with --die) when using --add-center"
        sd = options['add_noise']
        assert sd >= 0, f"The standard deviation cannot be negative: {sd}"
        random.seed(options['seed'])
        die = Die(options['die'])
        die_shape = Shape(die.width, die.height)

    if net_type == 'grid':
        data = gen_grid(size[0], size[1], 1, add_centers, sd, die_shape)
    elif net_type == 'chain':
        data = gen_chain(size[0], 1)
    elif net_type == 'ring':
        data = gen_ring(size[0], 1)
    elif net_type == 'star':
        data = gen_star(size[0], 1)
    elif net_type == 'ring-star':
        data = gen_ring_star(size[0], 1)
    elif net_type == 'one-net':
        data = gen_one_net(size[0], 1)
    elif net_type == 'htree':
        data = gen_htree(size[0], 1)
    else:
        assert False  # Should never happen

    yaml = YAML()
    yaml.default_flow_style = False
    with open(options['outfile'], 'w') as stream:
        yaml.dump(data, stream)
    return 0


def module_name(i: int, j: int = -1) -> str:
    """
    Creates the name of a module with subindices i,j. If j is negative, only one index is used
    :param i: first subindex
    :param j: second subindex
    :return: the name of the module
    """
    if j < 0:
        return "M%d" % i
    return "M%d_%d" % (i, j)


def gen_modules(area: float, rows: int, columns: int = 0,
                add_centers: bool = False, sd: float = 0, die_shape: Shape | None = None) \
        -> dict[str, Any]:
    """
    Creates a chain or a grid of modules. If columns is zero, only a chain is created
    :param area: area of each module
    :param rows: number of rows of the grid
    :param columns: number of columns of the grid (a chain if zero)
    :param add_centers: if True, centers are added to the modules (only supported for grids)
    :param sd: the standard deviation of the gaussian distribution used to add noise to the coordinates of the centers
    :param die_shape: the shape of the die used to calculate the centers when add_centers is True (else ignored)
    :return: the dictionary of modules
    """
    if columns <= 0:
        return {module_name(r): {KW.AREA: area} for r in range(rows)}

    modules: dict[str, dict[str, float | list[float]]] = \
        {module_name(r, c): {KW.AREA: area} for r in range(rows) for c in range(columns)}

    if add_centers:
        assert die_shape is not None
        x_offset = die_shape.w / columns
        y_offset = die_shape.h / rows
        for r in range(rows):
            for c in range(columns):
                modules[module_name(r, c)][KW.CENTER] = [(0.5 + c) * x_offset + random.gauss(0, sd),
                                                         (0.5 + r) * y_offset + random.gauss(0, sd)]

    return modules


def gen_grid(rows: int, columns: int, area: float,
             add_centers: bool = False, sd: float = 0, die_shape: Shape | None = None) \
        -> dict[str, Any]:
    """
    Generates the netlist of a grid
    :param rows: number of rows
    :param columns: number of columns
    :param area: area of each module
    :param add_centers: if True, centers are added to the modules
    :param sd: the standard deviation of the gaussian distribution used to add noise to the coordinates of the centers
    :param die_shape: the shape of the die used to calculate the centers when add_centers is True (else ignored)
    :return: a dictionary of the modules and the edges
    """
    modules = gen_modules(area, rows, columns, add_centers, sd, die_shape)
    horiz_edges = [[module_name(r, c), module_name(r, c + 1)] for r in range(rows) for c in range(columns - 1)]
    vert_edges = [[module_name(r, c), module_name(r + 1, c)] for r in range(rows - 1) for c in range(columns)]
    return {KW.MODULES: modules, KW.NETS: [*horiz_edges, *vert_edges]}


def gen_chain(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a chain
    :param n: number of modules
    :param area: area of each module
    :return: a dictionary of the modules and the edges
    """
    modules = gen_modules(area, n)
    edges = [[module_name(i), module_name(i + 1)] for i in range(n - 1)]
    return {KW.MODULES: modules, KW.NETS: edges}


def gen_ring(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a ring
    :param n: number of modules
    :param area: area of each module
    :return: a dictionary of the modules and the edges
    """
    modules = gen_modules(area, n)
    edges = [[module_name(i), module_name((i + 1) % n)] for i in range(n)]
    return {KW.MODULES: modules, KW.NETS: edges}


def gen_one_net(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a star with only one net that connects all nodes
    :param n: number of modules
    :param area: area of each module
    :return: a dictionary of the modules and the edges
    """
    modules = gen_modules(area, n)
    edges = [[module_name(i) for i in range(n)]]
    return {KW.MODULES: modules, KW.NETS: edges}


def gen_star(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a star with one node in the middle
    :param n: total number of modules
    :param area: area of each module
    :return: a dictionary of the modules and the edges
    """
    modules = gen_modules(area, n)
    edges = [[module_name(0), module_name(i)] for i in range(1, n)]
    return {KW.MODULES: modules, KW.NETS: edges}


def gen_ring_star(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a ring, including one node in the middle connected to all the other nodes
    :param n: total number of modules
    :param area: area of each module
    :return: a dictionary of the modules and the edges
    """
    modules = gen_modules(area, n)
    edges_ring = [[module_name(i), module_name(i + 1)] for i in range(1, n - 1)] + [
        [module_name(n - 1), module_name(1)]]
    edge_star = [[module_name(0), module_name(i)] for i in range(1, n)]
    return {KW.MODULES: modules, KW.NETS: edges_ring + edge_star}


def gen_htree(nlevels: int, area: float) -> dict[str, Any]:
    """
    Generates a H-tree
    :param nlevels: number of levels of the qtree
    :param area: area of each module
    :return: a dictionary of the modules and the edges
    """
    modules, edges, i = gen_htree_rec(nlevels, area, 1.0, 0)
    return {KW.MODULES: modules, KW.NETS: edges}


def gen_htree_rec(nlevels: int, area: float, weight: float, first_module: int)\
        -> tuple[dict[str, Any], list[list[str | float]], int]:
    """
    Generates a H-tree
    :param nlevels: number of levels of the htree
    :param area: area of each module
    :param weight: weight of the edges
    :param first_module: index of the first_module 
    :return: the dictionary of modules, the set of edges and the next available index for modules
    """
    assert nlevels > 0
    name_center = module_name(first_module)
    modules = {name_center: {KW.AREA: area}}
    edges: list[list[str | float]] = []
    if nlevels == 1:
        return modules, [], first_module + 1

    name_left, name_right = module_name(first_module + 1), module_name(first_module + 2)
    modules[name_left] = {KW.AREA: area}
    modules[name_right] = {KW.AREA: area}
    edges.append([name_left, name_center, weight])
    edges.append([name_right, name_center, weight])

    i = first_module + 3
    centers: list[int] = []
    for _ in range(4):
        centers.append(i)
        edges.append([module_name(first_module), module_name(i), weight])
        m, e, i = gen_htree_rec(nlevels - 1, area, 2*weight, i)
        modules.update(m)
        edges.extend(e)
    edges.append([name_left, module_name(centers[0]), weight])
    edges.append([name_left, module_name(centers[1]), weight])
    edges.append([name_right, module_name(centers[2]), weight])
    edges.append([name_right, module_name(centers[3]), weight])
    return modules, edges, i


if __name__ == "__main__":
    main()
