# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Tool for spectral floorplan. The algorithm implemented in this tool is based on
the one for Spectral Drawing proposed by Yehuda Koren in his paper
'Drawing Graphs by Eigenvectors: Theory and Practice'.
The algorithm has been modified to incorporate the mass of each node.
The mass is interpreted as the multiplicity of the node.
"""

import argparse
from itertools import combinations
import math
from typing import Any

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape
from frame.netlist.netlist import Netlist
from frame.utils.utils import Vector, Matrix
from tools.spectral.spectral_types import AdjEdge, AdjList
from tools.spectral.spectral_algorithm import spectral_layout_die


class Spectral(Netlist):
    """
    Class to extend a netlist with information for spectral placement
    """

    _adj: AdjList  # Adjacency list
    _mass: Vector  # Mass (size) of each node in _G
    _centers: (
        Matrix  # A 2xn matrix with the coordinates of the centers (negative if unknown)
    )
    _fixed_modules: list[bool]  # A boolean vector to indicate the fixed modules

    def __init__(self, stream: str):
        """
        Constructor
        :param stream: input stream from which the netlist is read
        """
        Netlist.__init__(self, stream)
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Creates the associated graph of the netlist. For hyper-edges with more than two pins, an extra node is created
        (with zero area) that is the center of the star. These nodes have a special attribute (hyper-node) to indicate
        whether the node is an original node (False) or the center of a hyper-edge (True).
        """

        # Dictionary: modulename -> index
        name2index = {m.name: i for i, m in enumerate(self.modules)}

        # Number of nodes, including the centers of the hyperedges with more than two nodes
        n = self.num_modules

        # Masses (0 for the centers of hyperedges)
        self._mass = [m.area() for m in self.modules]

        # Find the fixed modules
        self._fixed_modules = [False] * n
        self._centers = [[-1.0] * n, [-1.0] * n]

        # Centers
        for i, m in enumerate(self.modules):
            if m.center is not None:
                self._centers[0][i], self._centers[1][i] = m.center
            if m.is_fixed:
                assert m.center is not None
                self._fixed_modules[i] = True

        # Let us now create the adjacency list
        self._adj = [[] for _ in range(n)]  # Adjacency list (list of lists)

        # Clique model
        for e in self.edges:
            assert len(e.modules) > 1
            weight = 2 * e.weight / len(e.modules)
            for m1, m2 in combinations(e.modules, 2):
                src, dst = name2index[m1.name], name2index[m2.name]
                self._adj[src].append(AdjEdge(dst, weight))
                self._adj[dst].append(AdjEdge(src, weight))

    def spectral_layout(self, shape: Shape, nfloorplans: int, verbose: bool) -> int:
        """
        Computes a spectral layout of a graph in the rectangle with ll=(0,0).
        It defines the center of the modules. The algorithm is executed several times with different random
        initial points and the solution with best wirelength is selected. In case nfloorplans is zero, the
        initial location of the modules is used and only one floorplan is generated
        :param shape: shape of the die (width and height)
        :param nfloorplans: number of generated floorplans (if 0, initial centers are defined)
        :param verbose: indicates whether some verbose information must be printed
        """
        n = len(self._mass)
        assert n > 2, "Graph too small. Spectral layout needs more than 2 nodes."
        best_wl = math.inf
        best_coord = None
        if verbose:
            print("Spectral: verbose information")

        assert nfloorplans >= 0
        if nfloorplans == 0:
            nfloorplans = 1
            # assert all modules have centers
            for m in self.modules:
                assert m.center is not None, f"Module {m.name} has no initial center"
        else:
            # Remove centers of the non-fixed nodes
            for i in range(n):
                if not self._fixed_modules[i]:
                    self._centers[0][i], self._centers[1][i] = -1.0, -1.0

        for i in range(nfloorplans):
            coord, wl, niter = spectral_layout_die(
                self._adj,
                self._mass,
                [shape.w, shape.h],
                self._centers,
                self._fixed_modules,
            )
            if verbose:
                print(
                    "{:3d}:".format(i),
                    "  WL =",
                    "{:7.3f}".format(wl),
                    ", iterations(x,y) = (",
                    niter[0],
                    ",",
                    niter[1],
                    ")",
                    sep="",
                )
            if wl < best_wl:
                best_coord = coord
                best_wl = wl

        assert best_coord is not None

        for i, m in enumerate(self.modules):
            m.center = Point(
                best_coord[0][i] + shape.w / 2, best_coord[1][i] + shape.h / 2
            )

        # Reallocate the rectangles of hard modules and remove the center of hard and fixed modules
        # Still, the center is kept in case of terminals
        for m in self.modules:
            if m.is_hard:
                if not m.is_fixed:  # it is a hard module (movable)
                    m.recenter_rectangles()
                # Remove the center of the hard block
                if not m.is_iopin:  # Need to keep the center for the terminals
                    m.center = None

        return 0


def parse_options(
    prog: str | None = None, args: list[str] | None = None
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
        description="Compute the initial location for each module of the netlist using a "
        "combination of spectral and force-directed methods.",
    )
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-d",
        "--die",
        metavar="<WIDTH>x<HEIGHT> or FILENAME",
        help="size of the die (width x height) or name of the file",
    )
    parser.add_argument(
        "-i", "--init", action="store_true", help="use initial coordinates"
    )
    parser.add_argument(
        "--bestof",
        type=int,
        default=5,
        help="number of floorplans generated to select the best. Default: 5",
    )
    parser.add_argument("-o", "--outfile", required=True, help="output file (netlist)")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """Main function"""
    options = parse_options(prog, args)

    # Die
    die_file = options["die"]
    if die_file is not None:
        d = Die(die_file)
        die = Shape(d.width, d.height)
    else:
        die = Shape(1, 1)

    infile = options["netlist"]
    netlist = Spectral(infile)
    nfloorplans = options["bestof"]
    initial_center = options["init"]
    if initial_center:
        nfloorplans = 0
    assert initial_center or nfloorplans > 0, (
        "The number of floorplans must be a positive integer"
    )
    status = netlist.spectral_layout(die, nfloorplans, options["verbose"])
    netlist.write_yaml(options["outfile"])
    return status


if __name__ == "__main__":
    main()
