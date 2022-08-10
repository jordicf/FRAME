"""A netlist generator."""

import argparse
from typing import Any
from ruamel.yaml import YAML
from frame.utils.keywords import KW_MODULES, KW_NETS, KW_AREA


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, description="A netlist generator.", usage='%(prog)s [options]')
    parser.add_argument("-o", "--outfile", help="output file (netlist)", required=True)
    parser.add_argument("--type", type=str, choices=['grid', 'chain', 'ring', 'star'], required=True,
                        help="type of netlist (grid, chain, ring, star)")
    parser.add_argument("--size", type=int, nargs='+', required=True, help="size of the netlist")
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

    if net_type == 'grid':
        data = gen_grid(size[0], size[1], 1)
    elif net_type == 'chain':
        data = gen_chain(size[0], 1)
    elif net_type == 'ring':
        data = gen_ring(size[0], 1)
    elif net_type == 'star':
        data = gen_star(size[0], 1)
    else:
        assert False  # Should nver happen

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


def gen_modules(area: float, rows: int, columns: int = 0) -> dict[str, Any]:
    """
    Creates a chain or a grid of modules. If columns is zero, only a chain is created
    :param area: area of each module
    :param rows: number of rows of the grid
    :param columns: number of columns of the grid (a chain if zero)
    :return: the dictionary of modules
    """
    if columns <= 0:
        return {module_name(r): {KW_AREA: area} for r in range(rows)}
    return {module_name(r, c): {KW_AREA: area} for r in range(rows) for c in range(columns)}


def gen_grid(rows: int, columns: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a grid
    :param rows: number of rows
    :param columns: number of columns 
    :param area: area of each module
    :return: a dictionary of the modules
    """
    modules = gen_modules(area, rows, columns)
    horiz_edges = [[module_name(r, c), module_name(r, c + 1)] for r in range(rows) for c in range(columns - 1)]
    vert_edges = [[module_name(r, c), module_name(r + 1, c)] for r in range(rows - 1) for c in range(columns)]
    return {KW_MODULES: modules, KW_NETS: [*horiz_edges, *vert_edges]}


def gen_chain(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a chain
    :param n: number of modules
    :param area: area of each module
    :return: a dictionary of the modules
    """
    modules = gen_modules(area, n)
    edges = [[module_name(i), module_name(i + 1)] for i in range(n - 1)]
    return {KW_MODULES: modules, KW_NETS: edges}


def gen_ring(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a ring
    :param n: number of modules
    :param area: area of each module
    :return: a dictionary of the modules
    """
    modules = gen_modules(area, n)
    edges = [[module_name(i), module_name((i + 1) % n)] for i in range(n)]
    return {KW_MODULES: modules, KW_NETS: edges}


def gen_star(n: int, area: float) -> dict[str, Any]:
    """
    Generates the netlist of a star with only one net
    :param n: number of modules
    :param area: area of each module
    :return: a dictionary of the modules
    """
    modules = gen_modules(area, n)
    edges = [[module_name(i) for i in range(n)]]
    return {KW_MODULES: modules, KW_NETS: edges}


if __name__ == "__main__":
    main()
