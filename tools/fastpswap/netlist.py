# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from dataclasses import dataclass
from frame.netlist.netlist import Netlist
from frame.netlist.module import Module


@dataclass(slots=True)
class swapPoint:
    """A point in the netlist representing the position of a module."""

    x: float
    y: float
    nets: list[int]  # List of net IDs that this point belongs to (sorted and unique)

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.nets = []


@dataclass(slots=True)
class swapNet:
    """A net in the netlist."""

    weight: float  # Weight of the net
    points: list[int]  # List of point IDs that belong to this net
    hpwl: float  # Half-perimeter wire length (initialized to 0.0)

    def __init__(self, weight: float, points: list[int]) -> None:
        self.weight = weight
        self.points = points
        self.hpwl = 0.0


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
        "_avg_area",
        "_num_subblocks",
        "_split_net_factor",
        "_verbose",
    )

    _netlist: Netlist  # Original netlist
    _points: list[swapPoint]
    _nets: list[swapNet]
    _name2idx: dict[str, int]  # Mapping from names to indices of points
    _idx2name: list[str]  # Mapping from indices to names of points
    _movable: list[int]  # List of movable point indices
    _hpwl: float  # Total HPWL of the netlist
    _avg_area: float  # Average area of the movable modules
    _num_subblocks: int  # Number of fake subblocks created by splits
    _split_net_factor: float  # Weight factor for nets created by splits
    _verbose: bool  # Verbosity flag

    def __init__(
        self, filename: str, split_net_factor: float = 1.0, verbose: bool = False
    ) -> None:
        self._points = []
        self._nets = []
        self._name2idx = {}
        self._idx2name = []
        self._movable = []
        self._netlist = Netlist(filename)
        self._avg_area = 0.0
        self._num_subblocks = 0
        self._split_net_factor = split_net_factor
        self._verbose = verbose

        # Read modules
        self._netlist.calculate_centers_from_rectangles()
        movable_area = 0.0
        for m in self.netlist.modules:
            idx = len(self.points)
            self._name2idx[m.name] = idx
            self._idx2name.append(m.name)
            assert m.center is not None
            area = sum(r.area for r in m.rectangles)
            self.points.append(swapPoint(x=m.center.x, y=m.center.y))
            if not m.is_hard:
                assert (
                    len(m.rectangles) == 1
                ), "Only one rectangle per soft module is supported"
                self._movable.append(idx)
                movable_area += area

        assert len(self._movable) > 0, "No movable modules found"
        self._avg_area = movable_area / len(self._movable)

        # Read nets
        for e in self._netlist.edges:
            point_indices = []
            for m in e.modules:
                idx = self._name2idx[m.name]
                point_indices.append(idx)
                self.points[idx].nets.append(len(self.nets))
            self.nets.append(swapNet(weight=e.weight, points=point_indices))

        # Split movable modules that are too large
        if self._split_net_factor > 0.0:
            self._split_modules()

        # Sort the nets of each point
        for p in self.points:
            # Sorted and unique nets (to avoid repeated nets)
            p.nets = sorted(set(p.nets))
            
        # Compute initial HPWL
        self.hpwl = sum(self._compute_net_hpwl(net) for net in self.nets)

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

    def idx2module(self, i: int) -> Module:
        """Get the module corresponding to a point index."""
        name = self.idx2name(i)
        return self._netlist.get_module(name)

    def name2idx(self, name: str) -> int:
        """Mapping from module name to point index."""
        return self._name2idx[name]

    def _compute_net_hpwl(self, net: swapNet) -> float:
        """Compute the half-perimeter wire length (HPWL) of a net.
        It returns the computed HPWL."""
        xs = [self.points[p].x for p in net.points]
        ys = [self.points[p].y for p in net.points]
        net.hpwl = (max(xs) - min(xs) + max(ys) - min(ys)) * net.weight
        return net.hpwl
    
    def compute_total_hpwl(self) -> float:
        """Compute the total half-perimeter wire length (HPWL) of the netlist.
        It returns the computed total HPWL."""
        self.hpwl = sum(self._compute_net_hpwl(net) for net in self.nets)  # Reset HPWL
        return self.hpwl

    def _split_modules(self) -> None:
        """Split all movable modules that are too large."""
        num_movable = len(self.movable)
        for i in range(num_movable):
            idx = self.movable[i]
            m = self.idx2module(idx)
            assert not m.is_hard, "Only soft modules can be split"
            assert (
                len(m.rectangles) == 1
            ), "Only one rectangle per soft module is supported"
            r = m.rectangles[0]
            nrows, ncols = _best_split(
                r.shape.w, r.shape.h, self._avg_area, aspect_ratio=0.5
            )
            if nrows > 1 or ncols > 1:
                if self._verbose:
                    print(
                        f"Splitting module {m.name} (area {r.area:.1f}) "
                        f"into {nrows}x{ncols} sub-blocks "
                        f"with net weight {self._split_net_factor:.1f}"
                    )
                self._split_module(idx, nrows, ncols, net_weight=self._split_net_factor)

    def _split_module(
        self, idx: int, nrows: int, ncols: int, net_weight: float = 1.0
    ) -> None:
        """Split a module into a matrix of smaller modules."""
        assert nrows % 2 == 1 and ncols % 2 == 1, "Only odd splits are supported"
        # Get the rectangle of the module to be split
        m = self._netlist.get_module(self.idx2name(idx))
        assert not m.is_hard, "Only soft modules can be split"
        assert len(m.rectangles) == 1, "Only one rectangle per soft module is supported"
        r = m.rectangles[0]
        # Compute the new dimensions
        new_width = r.shape.w / ncols
        new_height = r.shape.h / nrows
        range_cols = ncols // 2
        range_rows = nrows // 2
        # Create new modules
        module = self.points[idx]
        for i in range(-range_cols, range_cols + 1):
            for j in range(-range_rows, range_rows + 1):
                if i == j == 0:
                    continue  # Skip the original module
                new_module = swapPoint(
                    x=module.x + i * new_width, y=module.y + j * new_height
                )
                new_idx = len(self.points)
                new_name = f"{m.name}_split_{new_idx}"
                self._name2idx[new_name] = new_idx
                self._idx2name.append(new_name)
                self.points.append(new_module)
                self.movable.append(new_idx)
                self._num_subblocks += 1
                # Star model for the nets between center and sub-blocks
                net_idx = len(self.nets)
                self.nets.append(swapNet(weight=net_weight, points=[idx, new_idx]))
                self.points[idx].nets.append(net_idx)
                self.points[new_idx].nets.append(net_idx)

    def remove_subblocks(self) -> None:
        """Remove all fake subblocks created by splits.
        Only the name mappings are removed."""
        if self._num_subblocks == 0:
            return

        if self._verbose:
            print(f"Removing {self._num_subblocks} fake subblocks created by splits.")
        for _ in range(self._num_subblocks):
            # Remove from idx2name mappings
            name = self._idx2name.pop()
            del self._name2idx[name]

        del self._points[-self._num_subblocks :]
        del self._movable[-self._num_subblocks :]
        del self._nets[-self._num_subblocks :]


def _best_split(
    width: float, height: float, target_area: float, aspect_ratio: float = 0.5
) -> tuple[int, int]:
    """Compute the best odd split (nrows, ncols) for a module of given
    width, height, and target area. The solution must satisfy the aspect ratio constraint.
    It returns the best (nrows, ncols) split."""
    area = width * height
    best_cost = float("inf")
    best_split = None
    max_slices = int(area / target_area) + 2
    for nrows in range(1, max_slices, 2):
        for ncols in range(1, max_slices, 2):
            new_width = width / ncols
            new_height = height / nrows
            block_area = new_width * new_height
            ratio = min(new_width / new_height, new_height / new_width)
            if ratio < aspect_ratio:
                continue
            cost = abs(target_area - block_area)
            if cost < best_cost:
                best_cost, best_split = cost, (nrows, ncols)

    if best_split is not None:
        return best_split
    # Just in case, try with a more relaxed aspect ratio
    return _best_split(width, height, target_area, aspect_ratio * 0.95)


if __name__ == "__main__":
    w = input()
    h = input()
    a = input()
    print(_best_split(float(w), float(h), float(a)))
