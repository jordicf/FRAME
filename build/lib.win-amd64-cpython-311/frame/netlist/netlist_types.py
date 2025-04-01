# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Types used in the netlist
"""

from dataclasses import dataclass
from math import sqrt

from .module import Module
from ..geometry.geometry import Point


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

    @property
    def wire_length(self) -> float:
        """
        Returns the wire length of the hyperedge.
        The wire length is the distance between the center of each module and the centroid of the
        modules (without taking into account module areas) multiplied by the hyperedge weight.
        """
        intersection_point = Point(0, 0)
        for b in self.modules:
            assert b.center is not None, "Module center must be defined. " \
                                         "Maybe execute b.calculate_center_from_rectangles()?"
            intersection_point += b.center
        intersection_point /= len(self.modules)
        wire_length = 0.0
        for b in self.modules:
            assert b.center is not None  # repeated assertion (will never happen) to suppress Mypy error
            v = intersection_point - b.center
            wire_length += sqrt(v & v)
        return wire_length * self.weight

    def __repr__(self) -> str:
        names = [b.name for b in self.modules]
        if self.weight == 1:
            return f'Edge<modules={names}>'
        return f'Edge<modules={names}, weight={self.weight}>'
