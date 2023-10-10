# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import argparse
import random
import os
from time import time
from typing import Any

from frame.die.die import Die
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point, NPoint

from tools.force.fruchterman_reingold import force_algorithm
from tools.force.multidimensional_shrink import force_algorithm as multiDim_force_algorithm
# from tools.force.fruchterman_reingold2 import force_algorithm as flexRepel_force_algorithm
from tools.force.kamada_kawai import kamada_kawai_layout

from tools.force.netlists.adjlist.benchmark import metrics

from tools.draw.draw import get_graph_plot


def add_noise(die: Die, sd: float = 0.01) -> Die:
    """
    Add some noise to the positions of the modules
    :param die: the die
    :param sd: standard deviation of the noise
    :return: the die with the noise added
    """
    assert die.netlist is not None, "No netlist associated to the die"  # Assertion to suppress Mypy error
    for module in die.netlist.modules:
        assert module.center is not None, "Module has no center"  # Assertion to suppress Mypy error
        if isinstance(module.center, Point):
            module.center.x += random.gauss(0, sd)
            module.center.y += random.gauss(0, sd)
        elif isinstance(module.center, NPoint):
            for dim in range(module.center.n):
                module.center.x[dim] += random.gauss(0, sd)
    return die


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog,
                                     description="Relocate the modules of the netlist using a force-directed algorithm "
                                                 "to obtain better initial values for the next stages.")
    parser.add_argument("--netlist", required=True,
                        help="input netlist filename")
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME", required=True,
                        help="size of the die (width x height) or name of the file")
    parser.add_argument("--heuristic", required=True, choices=['0','1','2'], help="heuristic for the force algorithm. \n 0 - FRAME\n1 - Multi Dimensional\n2 - FlexRepel")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the optimization logs, elapsed time, and some additional information")
    parser.add_argument("--visualize",
                        help="name of the GIF to visualize the full optimizations process "
                             "(if not present, no visualization is produced)")
    parser.add_argument("--out-netlist",
                        help="output netlist file (if not present, no file is produced)")
    parser.add_argument("-p", "--parallelize", action="store_true",
                        help="parallelize execution")
    parser.add_argument("-n", "--num-dimensions",
                        help="number of dimensions in multi heuristic force algorithm")
    parser.add_argument("-it", "--iterations",
                    help="number of iterations")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    random.seed(1234) # TODO: revisar

    netlist = Netlist(options["netlist"])

    for module in netlist.modules:
        if module.center is None: # TODO: revisar
            module.center = Point(random.gauss(0, 0.01), random.gauss(0, 0.01))

    die = Die(options["die"], netlist)

    verbose: bool = options["verbose"]
    visualize: str | None = options["visualize"]

    start_time = 0.0
    if verbose:
        start_time = time()

    die = add_noise(die)

    if options["iterations"] is not None:
        max_iter = int(options["iterations"])
    else:
        max_iter = 50

    if visualize:
        os.system("rm /tmp/FRAME-*")

    if options["num_dimensions"] is not None:
        dimensions = int(options["num_dimensions"])
    else:
        dimensions = 2

    if options["heuristic"] == "1":
        die, ma_vis_imgs = multiDim_force_algorithm(die, verbose=verbose, visualize=visualize, max_iter=max_iter,
                                       num_dimensions=dimensions, parallelize=options["parallelize"])
    elif options["heuristic"] == "2":
        die, ma_vis_imgs = flexRepel_force_algorithm(die, verbose=verbose, visualize=visualize, max_iter=max_iter)
    else:
        die, kk_vis_imgs = kamada_kawai_layout(die, verbose=verbose, visualize=visualize)
        die, f_vis_imgs = force_algorithm(die, verbose=verbose, visualize=visualize, max_iter=max_iter)
        ma_vis_imgs = kk_vis_imgs + f_vis_imgs

    for module in die.netlist.modules:
        module.center.x = float(module.center.x)
        module.center.y = float(module.center.y)

    if visualize:
        if options["heuristic"] == "1":
            framerate = max(1, min(5, max_iter//15))
            os.system(f"ffmpeg -loglevel quiet -r {framerate} -i /tmp/FRAME-tmp-image-%01d.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p -y movie.mp4")
        else:
            vis_imgs = ma_vis_imgs
            vis_imgs[0].save(f"{visualize}.gif", save_all=True, append_images=vis_imgs[1:], duration=100)

    if verbose:
        print(f"Elapsed time: {time() - start_time:.3f} s")
        die.netlist.write_yaml("/tmp/FRAME-tmp-output-netlist.yaml")
        for metric in metrics:
            print(metric.__name__.ljust(30), str(metric("/tmp/FRAME-tmp-output-netlist")).ljust(16))

    out_netlist_file: str | None = options["out_netlist"]
    if out_netlist_file is not None:
        assert die.netlist is not None, "No netlist associated to the die"  # Assertion to suppress Mypy error
        die.netlist.write_yaml(out_netlist_file)


if __name__ == "__main__":
    main()
