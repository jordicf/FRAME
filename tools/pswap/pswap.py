# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""Tool to explore different module locations by swapping their centroids"""

import argparse
import pathlib
from typing import Any, Optional
from .netlist import swapNetlist
from .anneal import simulated_annealing


def parse_options(
    prog: Optional[str] = None, args: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        usage="%(prog)s [options]",
        description="Explores different initial locations for the centroids "
        "of the modules using simulated annealing. It returns the best found solution.",
    )
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("-o", "--outfile", required=True, help="output file")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="number of iterations without improvement. Default: 10",
    )
    parser.add_argument(
        "--swaps",
        type=int,
        default=100,
        help="number of swaps per iteration and movable point. Default: 100",
    )
    parser.add_argument(
        "--accept",
        type=float,
        default=0.2,
        help="initial acceptance probability for new solutions. Default: 0.2",
    )
    parser.add_argument(
        "--tfactor",
        type=float,
        default=0.95,
        help="decreasing temperature factor. Default: 0.95",
    )

    return vars(parser.parse_args(args))


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> None:
    """Main function."""
    options = parse_options(prog, args)
    netlist = swapNetlist(options["netlist"])
    simulated_annealing(
        netlist,
        n_swaps=options["swaps"],
        patience=options["patience"],
        target_acceptance=options["accept"],
        temp_factor=options["tfactor"],
        verbose=options["verbose"]
    )
    netlist.netlist.update_centers(
        {netlist.idx2name(i): (p.x, p.y) for i, p in enumerate(netlist.points)}
    )
    if options["verbose"]:
        print(f"Final HPWL: {netlist.hpwl:.2f}")

    # Check the type of file by suffix
    outfile = options["outfile"]
    suffix = pathlib.Path(outfile).suffix

    if suffix == ".json":
        netlist.netlist.write_json(outfile)
    elif suffix in [".yaml", ".yml"]:
        netlist.netlist.write_yaml(options["outfile"])
    else:
        raise NameError(f"Unknown suffix for {outfile}: .json, .yaml or .yml expected")


if __name__ == "__main__":
    main()
