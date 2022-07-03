"""
Module to represent netlists
"""

import networkx as nx
from typing import TextIO
from io import StringIO

from keywords import KW_BLOCKS, KW_EDGES
from block import Block
from geometry import Rectangle
from netlist_types import Edge, HyperEdge
from yaml_read_netlist import parse_yaml_netlist
from yaml_write_netlist import dump_yaml_blocks, dump_yaml_edges
from ruamel.yaml import YAML


AdjList = list[list[Edge]]


class Netlist:
    """
    Class to represent a netlist
    """

    _blocks: list[Block]  # List of blocks
    _edges: list[HyperEdge]  # List of edges, with references to blocks
    _rectangles: list[Rectangle]  # List of rectangles
    _name2block: dict[str, Block]  # Map from block names to blocks
    _G: nx.Graph  # A graph representation of the netlist (star model for hyperedges with more than 2 pins)
    _adj: AdjList  # Adjacency list of _G (edge weights)
    _mass: list[float]  # Mass (size) of each node in _G

    def __init__(self, stream: str | TextIO, from_text: bool = False):
        """
        Constructor of a netlist from a file or from a string of text
        :param stream: name of the YAML file (str) or handle to the file
        :param from_text: if asserted, the stream is simply a text (not a file).
        """

        self._blocks, _named_edges = parse_yaml_netlist(stream, from_text)
        self._name2block = {b.name: b for b in self._blocks}

        # Edges
        self._edges = []
        for e in _named_edges:
            blocks = []
            for b in e.blocks:
                assert b in self._name2block, f'Unknown block {b} in edge'
                blocks.append(self._name2block[b])
            assert e.weight > 0, f'Incorrect edge weight {e.weight}'
            self._edges.append(HyperEdge(blocks, e.weight))

        # Create rectangles
        self._rectangles = [r for b in self.blocks for r in b.rectangles]

        # Create graph
        self._build_graph()

    @property
    def num_blocks(self) -> int:
        """Number of blocks of the netlist"""
        return len(self._blocks)

    @property
    def blocks(self) -> list[Block]:
        """List of blocks of the netlist"""
        return self._blocks

    @property
    def edges(self) -> list[HyperEdge]:
        """List of hyperedges of the netlist"""
        return self._edges

    @property
    def num_rectangles(self) -> int:
        """Number of rectangles of all blocks of the netlist"""
        return len(self.rectangles)

    @property
    def rectangles(self) -> list[Rectangle]:
        """Rectangles of all blocks of the netlist"""
        return self._rectangles

    @property
    def adjacency(self) -> AdjList:
        """Adjacency list of the netlist (including fake nodes for hyperedges)"""
        return self._adj

    @property
    def block_sizes(self) -> list[float]:
        """List of block sizes"""
        return self._mass

    def create_squares(self) -> list[Block]:
        """
        Creates a default rectangle (square) for each block that has no rectangles
        :return: The list of blocks for which a square has been created.
        """
        blocks = []
        for b in self.blocks:
            if b.num_rectangles == 0:
                b.create_square()
                blocks.append(b)
        return blocks

    def dump_yaml_netlist(self, filename: str = None) -> None | str:
        """
        Writes the netlist into a YAML file. If no file name is given, a string with the yaml contents
        is returned
        :param filename: name of the output file
        """
        data = {
            KW_BLOCKS: dump_yaml_blocks(self.blocks),
            KW_EDGES: dump_yaml_edges(self.edges)
        }

        yaml = YAML()
        yaml.default_flow_style = False
        if filename is None:
            string_stream = StringIO()
            yaml.dump(data, string_stream)
            output_str: str = string_stream.getvalue()
            string_stream.close()
            return output_str
        with open(filename, 'w') as stream:
            yaml.dump(data, stream)

    def _build_graph(self) -> None:
        """
        Creates the associated graph of the netlist. For hyper-edges with more than two pins, an extra node is created
        (with zero area) that is the center of the star. These nodes have a special attribute (hyper-node) to indicate
        whether the node is an original node (False) or the center of a hyper-edge (True).
        """
        self._G = nx.Graph()
        node_id = 0
        # The real nodes
        for b in self.blocks:
            self._G.add_node(b.name, hypernode=False, id=node_id)
            node_id += 1

        fake_id = 0
        for e in self.edges:
            if len(e.blocks) == 2:  # Normal edge (2 pins)
                self._G.add_edge(e.blocks[0].name, e.blocks[1].name, weight=e.weight)
            else:  # Hyperedge (more than 2 pins)
                # Generate a name for the hypernode (not colliding with other nodes)
                while True:
                    fake_b = "_hyper_" + str(fake_id)
                    fake_id += 1
                    if fake_b not in self._name2block:
                        break
                # Create the center of the hyperedge
                self._G.add_node(fake_b, hypernode=True, id=node_id)
                node_id += 1
                # Add edges from the center to each node
                for b in e.blocks:
                    self._G.add_edge(fake_b, b.name, weight=e.weight)

        self._adj = [[] for _ in range(node_id)]  # Adjacency list (list of lists)
        self._mass = [0.0] * node_id  # List of masses (initially all zero)

        for name, block in self._name2block.items():
            ident = self._G.nodes[name]['id']
            self._mass[ident] = block.area()

        for b, nbrs in self._G.adj.items():
            idx = self._G.nodes[b]['id']
            adj = self._adj[idx]
            for nbr, attr in nbrs.items():
                adj.append(Edge(self._G.nodes[nbr]['id'], attr['weight']))
