"""
Module for spectral layout. The algorithm implemented in this module is based on the one for Spectral Drawing
proposed by Yehuda Koren in his paper 'Drawing Graphs by Eigenvectors: Theory and Practice'. The algorithm has been
modified to incorporate the mass of each node. The mass is interpreted as the multiplicity of the node.
"""
import networkx as nx
from argparse import ArgumentParser
from typing import TextIO, NamedTuple, Any

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape
from frame.netlist.netlist import Netlist
from spectral_algorithm import spectral_layout_unit_square


# Representation of an edge in the adjacency list of a graph (possibly obtained from a hypergraph).
# The edge represents the target node in an adjacency list
class AdjEdge(NamedTuple):
    node: int  # target node
    weight: float  # Weight of the edge


AdjList = list[list[AdjEdge]]
Vector = list[float]


class Spectral(Netlist):
    """
    Class to extend a netlist with information for spectral placement
    """

    _G: nx.Graph  # A graph representation of the netlist (star model for hyperedges with more than 2 pins)
    _adj: AdjList  # Adjacency list of _G (edge weights)
    _mass: Vector  # Mass (size) of each node in _G

    def __init__(self, stream: str | TextIO, from_text: bool = False):
        Netlist.__init__(self, stream, from_text)
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Creates the associated graph of the netlist. For hyper-edges with more than two pins, an extra node is created
        (with zero area) that is the center of the star. These nodes have a special attribute (hyper-node) to indicate
        whether the node is an original node (False) or the center of a hyper-edge (True).
        """
        self._G = nx.Graph()
        node_id = 0
        # The real nodes
        for m in self.modules:
            self._G.add_node(m.name, hypernode=False, id=node_id)
            node_id += 1

        # Fake nodes for centers of hyperedges
        fake_id = 0
        for e in self.edges:
            if len(e.modules) == 2:  # Normal edge (2 pins)
                # We use weight/2 to mimic the star model when using hyperedges
                self._G.add_edge(e.modules[0].name, e.modules[1].name, weight=e.weight / 2)
            else:  # Hyperedge (more than 2 pins)
                # Generate a name for the hypernode (not colliding with other nodes)
                while True:
                    fake_m = "_hyper_" + str(fake_id)
                    fake_id += 1
                    if fake_m not in self._name2module:
                        break
                # Create the center of the hyperedge
                self._G.add_node(fake_m, hypernode=True, id=node_id)
                node_id += 1
                # Add edges from the center to each node
                for m in e.modules:
                    self._G.add_edge(fake_m, m.name, weight=e.weight)

        self._adj = [[] for _ in range(node_id)]  # Adjacency list (list of lists)
        self._mass = [0.0] * node_id  # List of masses (initially all zero)

        for name, module in self._name2module.items():
            ident = self._G.nodes[name]['id']
            self._mass[ident] = module.area()

        for b, nbrs in self._G.adj.items():
            idx = self._G.nodes[b]['id']
            adj = self._adj[idx]
            for nbr, attr in nbrs.items():
                adj.append(AdjEdge(self._G.nodes[nbr]['id'], attr['weight']))

    def spectral_layout(self, shape: Shape) -> int:
        """
        Computes a spectral layout of a graph in the rectangle with ll=(0,0).
        It defines the center of the modules
        :param shape: shape of the die (width and height)
        """
        assert len(self._mass) > 2, "Graph too small. Spectral layout needs more than 2 nodes."
        coord = spectral_layout_unit_square(self._adj, self._mass, 3)
        # If width < height, swap the dimensions
        span_x = max(coord[0]) - min(coord[0])
        span_y = max(coord[1]) - min(coord[1])
        coordinates_wider = span_x > span_y
        layout_wider = shape.w > shape.h

        # Swap coordinates to have a smaller scaling
        if coordinates_wider != layout_wider:
            coord[0], coord[1] = coord[1], coord[0]

        # Define the centers of the modules
        scale_x, scale_y = shape.w / 2, shape.h / 2
        for i, m in enumerate(self.modules):
            m.center = Point((coord[0][i] + 1) * scale_x, (coord[1][i] + 1) * scale_y)

        return 0


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="A floorplan drawing tool", usage='%(prog)s [options]')
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("--die", help="Size of the die (width x height) or name of the file",
                        metavar="<width>x<height> or filename")
    parser.add_argument("-o", "--outfile", help="output file (netlist)")
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

    return netlist.spectral_layout(die)


if __name__ == "__main__":
    main()
