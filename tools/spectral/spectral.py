"""
Module for spectral floorplan. The algorithm implemented in this module is based on the one for Spectral Drawing
proposed by Yehuda Koren in his paper 'Drawing Graphs by Eigenvectors: Theory and Practice'. The algorithm has been
modified to incorporate the mass of each node. The mass is interpreted as the multiplicity of the node.
"""
import argparse
from typing import Any

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape
from frame.netlist.netlist import Netlist
from frame.utils.utils import Vector, TextIO_String
from tools.spectral.spectral_types import AdjEdge, AdjList
from tools.spectral.spectral_algorithm import spectral_layout_unit_square, scale_coordinates


class Spectral(Netlist):
    """
    Class to extend a netlist with information for spectral placement
    """

    _adj: AdjList  # Adjacency list
    _mass: Vector  # Mass (size) of each node in _G

    def __init__(self, stream: TextIO_String):
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
        self._mass = [0.0] * nnodes  # List of masses (initially all zero)
        for i, m in enumerate(self.modules):  # Only the masses of the non-fake nodes
            self._mass[i] = m.area()

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

    def spectral_layout(self, shape: Shape) -> int:
        """
        Computes a spectral layout of a graph in the rectangle with ll=(0,0).
        It defines the center of the modules
        :param shape: shape of the die (width and height)
        """
        assert len(self._mass) > 2, "Graph too small. Spectral layout needs more than 2 nodes."
        coord = spectral_layout_unit_square(self._adj, self._mass, 3)
        scale_coordinates(coord, self._mass, shape.w, shape.h)

        for i, m in enumerate(self.modules):
            m.center = Point(coord[0][i], coord[1][i])

        return 0


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, description="A floorplan drawing tool", usage='%(prog)s [options]')
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("-d", "--die", help="size of the die (width x height) or name of the file",
                        metavar="<width>x<height> or filename")
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
    status = netlist.spectral_layout(die)
    netlist.write_yaml(options['outfile'])
    return status


if __name__ == "__main__":
    main()
