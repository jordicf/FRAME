from typing import NamedTuple


# Representation of an edge in the adjacency list of a graph (possibly obtained from a hypergraph).
# The edge represents the target node in an adjacency list
class AdjEdge(NamedTuple):
    node: int  # target node
    weight: float  # Weight of the edge


# Type for the adjacency list
AdjList = list[list[AdjEdge]]
