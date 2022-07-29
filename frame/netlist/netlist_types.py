from dataclasses import dataclass
from .module import Module


# Representation of an edge in a graph (possibly obtained from a hypergraph).
# The edge represents the target node in an adjacency list
@dataclass()
class Edge:
    node: str  # Name of the target
    weight: float  # Weight of the edge


# Representation of a hyperedge (list of modules)
@dataclass()
class HyperEdge:
    modules: list[Module]  # List of modules of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        names = [b.name for b in self.modules]
        if self.weight == 1:
            return f'Edge<modules={names}>'
        return f'Edge<modules={self.modules}, weight={self.weight}>'


# Representation of a hyperedge (list of module names)
@dataclass()
class NamedHyperEdge:
    modules: list[str]  # List of module names of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        if self.weight == 1:
            return f'Edge<modules={self.modules}>'
        return f'Edge<modules={self.modules}, weight={self.weight}>'
