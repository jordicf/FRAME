# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from dataclasses import dataclass
from frame.netlist.netlist import Netlist


@dataclass(slots=True)
class swapPoint:
    """A point in the netlist."""

    x: float
    y: float
    nets: list[int]  # List of net IDs that this point belongs to


@dataclass(slots=True)
class swapNet:
    """A net in the netlist."""

    weight: float  # Weight of the net
    points: list[int]  # List of point IDs that belong to this net
    hpwl: float = 0.0  # Half-perimeter wire length (initialized to 0.0)


class swapNetlist:
    """A netlist consisting of points and nets."""

    __slots__ = (
        "_netlist",
        "_points",
        "_nets",
        "_name2idx",
        "_idx2name",
        "_movable",
        "_hpwl",
    )
    _netlist: Netlist  # Original netlist
    _points: list[swapPoint]
    _nets: list[swapNet]
    _name2idx: dict[str, int]  # Mapping from names to indices of points
    _idx2name: list[str]  # Mapping from indices to names of points
    _movable: list[int]  # List of movable point indices
    _hpwl: float  # Total HPWL of the netlist

    def __init__(self, filename: str) -> None:
        self._points = []
        self._nets = []
        self._name2idx = {}
        self._idx2name = []
        self._movable = []
        self._netlist = Netlist(filename)

        # Read modules
        self._netlist.calculate_centers_from_rectangles()
        for m in self._netlist.modules:
            idx = len(self.points)
            self._name2idx[m.name] = idx
            self._idx2name.append(m.name)
            assert m.center is not None
            self.points.append(swapPoint(x=m.center.x, y=m.center.y, nets=[]))
            if not m.is_hard:
                self._movable.append(idx)

        # Read nets
        for e in self._netlist.edges:
            point_indices = []
            for m in e.modules:
                idx = self._name2idx[m.name]
                point_indices.append(idx)
                self._points[idx].nets.append(len(self.nets))
            self.nets.append(swapNet(weight=e.weight, points=point_indices))

        self.hpwl = sum(self.compute_net_hpwl(net) for net in self.nets)

    @property
    def netlist(self) -> Netlist:
        """Original netlist."""
        return self._netlist

    @property
    def movable(self) -> list[int]:
        """List of movable point indices."""
        return self._movable

    @property
    def points(self) -> list[swapPoint]:
        """List of points in the netlist."""
        return self._points

    @property
    def nets(self) -> list[swapNet]:
        """List of nets in the netlist."""
        return self._nets

    @property
    def hpwl(self) -> float:
        """Total half-perimeter wire length (HPWL) of the netlist."""
        return self._hpwl

    @hpwl.setter
    def hpwl(self, value: float) -> None:
        self._hpwl = value

    def idx2name(self, i: int) -> str:
        """Mapping from point index to module name."""
        return self._idx2name[i]

    def compute_net_hpwl(self, net: swapNet) -> float:
        """Compute the half-perimeter wire length (HPWL) of a net.
        It returns the computed HPWL."""
        xs = [self.points[p].x for p in net.points]
        ys = [self.points[p].y for p in net.points]
        net.hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys)) * net.weight
        return net.hpwl

    def swap_points(self, idx1: int, idx2: int) -> float:
        """Swap two points and return the change in total HPWL."""
        p1, p2 = self.points[idx1], self.points[idx2]
        affected_nets = set(p1.nets) | set(p2.nets)
        p1.x, p2.x = p2.x, p1.x
        p1.y, p2.y = p2.y, p1.y
        delta_hpwl = -sum(self.nets[net_idx].hpwl for net_idx in affected_nets)
        delta_hpwl += sum(
            self.compute_net_hpwl(self.nets[net_idx]) for net_idx in affected_nets
        )
        self.hpwl += delta_hpwl
        return delta_hpwl
