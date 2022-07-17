from typing import Any
from argparse import ArgumentParser

def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="A netlist generator", usage='%(prog)s [options]')
    parser.add_argument("-o", "--outfile", help="output file (netlist)", required=True)
    parser.add_argument("--type", type=str, choices=['grid', 'chain', 'ring', 'star'], required=True,
                        help="type of netlist (grid, chain, ring, star)")
    parser.add_argument("--size", type=int, nargs='+', required=True, help="size of the netlist")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """Main function."""
    options = parse_options(prog, args)
    nsize = len(options['size'])
    type = options['type']
    if type == 'grid':
        assert nsize == 2, "Two parameters must be specified for the size of a grid"
    else:
        assert nsize == 1, "Too many parameters for the size of the netlist"
    print(options)
    return 0


if __name__ == "__main__":
    main()