"""
Tool for spectral floorplan. The algorithm implemented in this tool is based on the one for Spectral Drawing
proposed by Yehuda Koren in his paper 'Drawing Graphs by Eigenvectors: Theory and Practice'. The algorithm has been
modified to incorporate the mass of each node. The mass is interpreted as the multiplicity of the node.
"""
import argparse
import math
from typing import Any

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape
from frame.netlist.netlist import Netlist
from frame.utils.utils import Vector, Matrix, TextIO_String
from tools.spectral.spectral_types import AdjEdge, AdjList
from tools.spectral.spectral_algorithm import spectral_layout_unit_square


class Spectral(Netlist):
    """
    Class to extend a netlist with information for spectral placement
    """

    _adj: AdjList  # Adjacency list
    _mass: Vector  # Mass (size) of each node in _G
    _fixed_coord: Matrix  # A 2xn matrix with the fixed coordinates (negative if floating)

    def __init__(self, stream: TextIO_String):
        """Constructor"""
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
        nnodes = self.num_modules + sum(1 for e in self.edges if len(e.modules) > 2)
        # Masses (0 for the centers of hyperedges)
        self._mass = [m.area() for m in self.modules] + [0.0] * (nnodes - self.num_modules)

        # Find the fixed modules
        self._fixed_coord = [[-1.0] * nnodes, [-1.0] * nnodes]
        for i, m in enumerate(self.modules):
            if m.fixed:
                assert m.center is not None
                self._fixed_coord[0][i], self._fixed_coord[1][i] = m.center

        # Let us now create the adjacency list
        self._adj = [[] for _ in range(nnodes)]  # Adjacency list (list of lists)

        # Fake nodes for centers of hyperedges
        fake_node = self.num_modules  # The first fake node in the adjacency list
        for e in self.edges:
            if len(e.modules) == 2:  # Normal edge (2 pins)
                # We use weight/2 to mimic the star model when using hyperedges
                src, dst = name2index[e.modules[0].name], name2index[e.modules[1].name]
                self._adj[src].append(AdjEdge(dst, e.weight / 2))
                self._adj[dst].append(AdjEdge(src, e.weight / 2))
            else:  # Hyperedge (more than 2 pins)
                for m in e.modules:
                    idx = name2index[m.name]
                    self._adj[idx].append(AdjEdge(fake_node, e.weight))
                    self._adj[fake_node].append(AdjEdge(idx, e.weight))
                fake_node += 1

    def spectral_layout(self, shape: Shape, nfloorplans: int, verbose: bool) -> int:
        """
        Computes a spectral layout of a graph in the rectangle with ll=(0,0).
        It defines the center of the modules
        :param shape: shape of the die (width and height)
        :param nfloorplans: number of generated floorplans
        :param verbose: indicates whether some verbose information must be printed
        """
        assert len(self._mass) > 2, "Graph too small. Spectral layout needs more than 2 nodes."
        best_wl = math.inf
        best_coord = None
        if verbose:
            print("Spectral: verbose information")
        for i in range(nfloorplans):
            coord, wl, niter = spectral_layout_unit_square(self._adj, self._mass,
                                                           [shape.w, shape.h], 2, self._fixed_coord)
            if verbose:
                print("{:3d}:".format(i), "  WL =", "{:7.3f}".format(wl),
                      ", iterations(x,y) = (", niter[0], ",", niter[1], ")", sep='')
            if wl < best_wl:
                best_coord = coord
                best_wl = wl

        assert best_coord is not None

        for i, m in enumerate(self.modules):
            m.center = Point(best_coord[0][i] + shape.w / 2, best_coord[1][i] + shape.h / 2)

        return 0


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, usage='%(prog)s [options]',
                                     description="Compute the initial location for each module of the netlist using a "
                                                 "combination of spectral and force-directed methods.")
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME",
                        help="size of the die (width x height) or name of the file")
    parser.add_argument("--bestof", type=int, default=5,
                        help="number of floorplans generated to select the best. Default: 5")
    parser.add_argument("-o", "--outfile", required=True, help="output file (netlist)")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """Main function"""
    options = parse_options(prog, args)

    # Die
    die_file = options['die']
    if die_file is not None:
        d = Die(die_file)
        die = Shape(d.width, d.height)
    else:
        die = Shape(1, 1)

    infile = options['netlist']
    netlist = Spectral(infile)
    nfloorplans = options['bestof']
    assert nfloorplans > 0, "The number of floorplans must be a positive integer"
    status = netlist.spectral_layout(die, nfloorplans, options['verbose'])
    netlist.write_yaml(options['outfile'])
    return status


if __name__ == "__main__":
    main()
