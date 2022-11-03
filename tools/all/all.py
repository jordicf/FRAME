# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import argparse
import subprocess
import os
from time import time
from typing import Any


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, description="Run the whole FRAME flow.")
    parser.add_argument("--netlist", required=True,
                        help="input netlist filename")
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME", required=True,
                        help="size of the die (width x height) or name of the file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the optimization logs, elapsed time, and some additional information")
    return vars(parser.parse_args(args))


def execute_command(command: list[str], verbose: bool) -> None:
    """
    Execute a command and print the elapsed time if verbose is True
    :param command: the command to execute
    :param verbose: if True, print the elapsed time
    """
    start_time = 0.0
    if verbose:
        command.append("--verbose")
        print(f"Executing \"{' '.join(command)}\"")
        start_time = time()
    subprocess_info = subprocess.run(command, shell=True)
    if verbose:
        print(f"Finished! Elapsed time: {time() - start_time:.3f} s")
        print("-" * 80)
    if subprocess_info.returncode != 0:
        raise RuntimeError(f"\"{' '.join(command)}\" exited abnormally!")


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    verbose: bool = options["verbose"]
    start_time = 0.0
    if verbose:
        start_time = time()

    netlist = os.path.basename(options['netlist'])
    execute_command(["frame", "spectral",
                     netlist,
                     "--die", options['die'],
                     "--outfile", "spectral-" + netlist], verbose)
    execute_command(["frame", "force",
                     "--netlist", "spectral-" + netlist,
                     "--die", options['die'],
                     "--out-netlist", "force-" + netlist], verbose)
    execute_command(["frame", "glbfloor",
                     "--netlist", "force-" + netlist,
                     "--die", options['die'],
                     "--out-netlist", "glbfloor-" + netlist,
                     "--out-allocation", "glbfloor-allocation.yml",
                     "--max-iter", "2",
                     "--aspect-ratio", "2",
                     "--num-rectangles", "16"], verbose)
    execute_command(["frame", "rect",
                     "glbfloor-allocation.yml",
                     "--netlist", "glbfloor-" + netlist,
                     "--outfile", "rect-" + netlist], verbose)
    execute_command(["frame", "legalfloor",
                     "rect-" + netlist,
                     options['die'],
                     "--outfile", "legalfloor-" + netlist], verbose)

    if verbose:
        print(f"Total elapsed time: {time() - start_time:.3f} s")


if __name__ == "__main__":
    main()
