from dataclasses import dataclass

from .module import Module


@dataclass()
class Edge:
    """
    Representation of an edge in a graph (possibly obtained from a hypergraph).
    The edge represents the target node in an adjacency list
    """
    node: str  # Name of the target
    weight: float  # Weight of the edge


@dataclass()
class NamedHyperEdge:
    """Representation of a hyperedge (list of module names)"""
    modules: list[str]  # List of module names of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        if self.weight == 1:
            return f'Edge<modules={self.modules}>'
        return f'Edge<modules={self.modules}, weight={self.weight}>'


@dataclass()
class HyperEdge:
    """Representation of a hyperedge (list of modules)"""
    modules: list[Module]  # List of modules of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        names = [b.name for b in self.modules]
        if self.weight == 1:
            return f'Edge<modules={names}>'
        return f'Edge<modules={names}, weight={self.weight}>'
