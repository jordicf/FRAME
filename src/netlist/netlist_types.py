from typing import NamedTuple
from block import Block


# Representation of an edge in a graph (possibly obtained from a hypergraph).
# The edge represents the target node in an adjacency list
class Edge(NamedTuple):
    node: str  # Name of the target
    weight: float  # Weight of the edge


# Representation of a hyperedge (list of Blocks)
class HyperEdge(NamedTuple):
    blocks: list[Block]  # List of blocks of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        names = [b.name for b in self.blocks]
        if self.weight == 1:
            return f'Edge<blocks={names}>'
        return f'Edge<blocks={self.blocks}, weight={self.weight}>'


# Representation of a hyperedge (list of Block names)
class NamedHyperEdge(NamedTuple):
    blocks: list[str]  # List of block names of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        if self.weight == 1:
            return f'Edge<blocks={self.blocks}>'
        return f'Edge<blocks={self.blocks}, weight={self.weight}>'
