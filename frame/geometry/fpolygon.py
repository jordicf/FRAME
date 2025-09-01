# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"This module implements a polygon class based on the rportion library."

from __future__ import annotations
import copy
from typing import Iterable, Iterator, Optional
from dataclasses import dataclass
import itertools
from pprint import pprint
import portion as p
from rportion import RPolygon, rclosedopen

# Some auxiliary types
# Tuple to represent a rectangle in the interval [xmin, xmax, ymin, ymax]


@dataclass
class XY_Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self) -> float:
        """Returns the width of the box."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Returns the height of the box."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Returns the area of the box."""
        return self.width * self.height
    
    def dup(self) -> XY_Box:
        """Returns a duplication of the box."""
        return XY_Box(self.xmin, self.xmax, self.ymin, self.ymax)

    def __repr__(self) -> str:
        return f"[x=({self.xmin},{self.xmax}),y=({self.ymin},{self.ymax})]"

    def __hash__(self) -> int:
        return hash((self.xmin, self.xmax, self.ymin, self.ymax))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, XY_Box):
            return NotImplemented
        return (
            self.xmin == other.xmin
            and self.xmax == other.xmax
            and self.ymin == other.ymin
            and self.ymax == other.ymax
        )

    def bbox(self, other: XY_Box) -> XY_Box:
        """Returns the bounding box of self and other."""
        return XY_Box(
            xmin=min(self.xmin, other.xmin),
            xmax=max(self.xmax, other.xmax),
            ymin=min(self.ymin, other.ymin),
            ymax=max(self.ymax, other.ymax),
        )

    def distance(self, other: XY_Box) -> float:
        """Returns the Manhattan distance between the two rectangles."""
        dx = max(0.0, other.xmin - self.xmax, self.xmin - other.xmax)
        dy = max(0.0, other.ymin - self.ymax, self.ymin - other.ymax)
        return dx + dy

    def touch(self, other: XY_Box) -> bool:
        """Returns True if the two rectangles touch each other (i.e., they have
        at least one point in common)."""
        return (
            self.xmin <= other.xmax
            and self.xmax >= other.xmin
            and self.ymin <= other.ymax
            and self.ymax >= other.ymin
        )

    def iso_area_merge(self, other: XY_Box, direction: str) -> XY_Box:
        """Returns a rectangle with the same area as the sum of the two
        rectangles. The rectangle is aligned with the edge of the trunk
        corresponding to the direction (ymin for N, ymax for S, xmin for E, xmax for W).
        The center of gravity of the resulting rectangle is the same as the
        center of gravity of the two rectangles weighted by their area."""
        assert direction in "NSEW", f"Invalid direction {direction} in iso_area_merge"
        area = self.area + other.area
        self_ratio = self.area / area
        other_ratio = other.area / area
        xc = (
            (self.xmin + self.xmax) * self_ratio
            + (other.xmin + other.xmax) * other_ratio
        ) / 2
        yc = (
            (self.ymin + self.ymax) * self_ratio
            + (other.ymin + other.ymax) * other_ratio
        ) / 2
        if direction == "N":
            height = 2*(yc - self.ymin)
            width = area / height
        elif direction == "S":
            height = 2*(self.ymax - yc)
            width = area / height
        elif direction == "E":
            width = 2*(xc - self.xmin)
            height = area / width
        else:  # direction == "W"
            width = 2*(self.xmax - xc)
            height = area / width
        return XY_Box(xc - width/2, xc + width/2, yc - height/2, yc + height/2)

    def iso_area_absorb(self, other: XY_Box, direction: str) -> XY_Box:
        """Returns a rectangle with the same area as the sum of the two
        rectangles. The rectangle is aligned with the edge of the trunk
        corresponding to the direction (ymin for N, ymax for S, xmin for E, xmax for W).
        The biggest rectangle absorbs the smallest one by enlarging the corresponding edge
        without exceeding the boundary of the other rectangle."""
        assert direction in "NSEW", f"Invalid direction {direction} in iso_area_merge"
        # This function is tedious, since it has many cases (NSEW, left/right/top/bottom)
        pprint(f"Absorbing {self} and {other} in direction {direction}")

        big, small = (self, other) if self.area >= other.area else (other, self)
        new_b = big.dup()
        if direction in "NS":
            left = big.xmin < small.xmin
            if left: # big is to the left of small
                new_b.xmax += small.area/big.height
                if new_b.xmax > small.xmax:
                    inc_area = (new_b.xmax - small.xmax)*new_b.height
                    new_b.xmax = small.xmax
                    if direction == "N":
                        new_b.ymax += inc_area / new_b.width
                    else: # direction == "S"
                        new_b.ymin -= inc_area / new_b.width
            else: # big is to the right of small
                new_b.xmin -= small.area/big.height
                if new_b.xmin < small.xmin:
                    inc_area = (small.xmin - new_b.xmin)*new_b.height
                    new_b.xmin = small.xmin
                    if direction == "N":
                        new_b.ymax += inc_area / new_b.width
                    else: # direction == "S"
                        new_b.ymin -= inc_area / new_b.width
        else: # direction in "EW"
            top = big.ymin > small.ymin
            if top: # big is above small
                new_b.ymin -= small.area/big.width
                if new_b.ymin < small.ymin:
                    inc_area = (small.ymin - new_b.ymin)*new_b.width
                    new_b.ymin = small.ymin
                    if direction == "E":
                        new_b.xmax += inc_area / new_b.height
                    else: # direction == "W"
                        new_b.xmin -= inc_area / new_b.height
            else: # big is below small
                new_b.ymax += small.area/big.width
                if new_b.ymax > small.ymax:
                    inc_area = (new_b.ymax - small.ymax)*new_b.width
                    new_b.ymax = small.ymax
                    if direction == "E":
                        new_b.xmax += inc_area / new_b.height
                    else: # direction == "W"
                        new_b.xmin -= inc_area / new_b.height
        pprint(f"Result: {new_b}")
        return new_b
        
# Some methods to extend the functionality of rportion


def _RPoly2Box(r: RPolygon) -> XY_Box:
    """Converts a rectangle in RPolygon format to Rectangle format."""
    x_interval = r.x_enclosure_interval
    xmin, xmax = x_interval.lower, x_interval.upper
    y_interval = r.y_enclosure_interval
    ymin, ymax = y_interval.lower, y_interval.upper
    assert (
        isinstance(xmin, float)
        and isinstance(xmax, float)
        and isinstance(ymin, float)
        and isinstance(ymax, float)
    )
    return XY_Box(xmin, xmax, ymin, ymax)


def closest_boxes(boxes: Iterable[XY_Box]) -> tuple[XY_Box, XY_Box]:
    """Returns the pair of closest boxes in the given iterable."""
    min_dist = float("inf")
    closest_pair = (XY_Box(0, 0, 0, 0), XY_Box(0, 0, 0, 0))  # fake assignment
    for b1, b2 in itertools.combinations(boxes, 2):
        dist = b1.distance(b2)
        if dist < min_dist:
            min_dist = dist
            closest_pair = (b1, b2)
    assert min_dist < float("inf")
    return closest_pair


class StropDecomposition:
    """Class to represent the STROP decomposition of a polygon"""

    _ref: FPolygon
    _trunk: XY_Box
    _branches: dict[str, set[XY_Box]]  # str in 'N', 'S', 'E', 'W'
    _similarity: float  # Jaccard similarity wrt the original polygon

    def __init__(self, ref: FPolygon, trunk: XY_Box) -> None:
        self._ref = ref
        self._similarity = -1.0
        self._trunk = trunk
        self._branches = {"N": set(), "S": set(), "E": set(), "W": set()}

    def __eq__(self, other: object) -> bool:
        """Checks equality of two STROP decompositions"""
        if not isinstance(other, StropDecomposition):
            return NotImplemented
        return self._trunk == other._trunk and all(
            self._branches[d] == other._branches[d] for d in "NSEW"
        )

    def __hash__(self) -> int:
        """Returns a hash value for the STROP decomposition."""
        h = hash(self._trunk)
        for d in "NSEW":
            for b in self._branches[d]:
                h ^= hash(b)
        return h

    def dup(self) -> StropDecomposition:
        """Returns a duplication of the StropDecomposition."""
        c = StropDecomposition(self._ref, self._trunk)
        c._branches = copy.deepcopy(self._branches)
        return c

    def __repr__(self) -> str:
        s = f"T:{self._trunk}"
        for d in "NSEW":
            for b in self._branches[d]:
                s += f"|{d}:{b}"
        return s

    @property
    def trunk(self) -> XY_Box:
        """Returns the trunk of the decomposition."""
        return self._trunk

    @property
    def similarity(self) -> float:
        """Returns the Jaccard similarity of the decomposition with
        respect to the original polygon."""
        if self._similarity < 0.0:
            self._similarity = round(self._ref.jaccard_similarity(self.polygon), 8)
        return self._similarity

    def add_rectangle(self, r: XY_Box, direction: str) -> None:
        """Adds a rectangle to the decomposition."""
        if direction == "T":
            self._trunk = r
        else:
            self._branches[direction].add(r)
        self._similarity = -1.0  # Invalidate similarity

    @property
    def num_branches(self) -> int:
        """Returns the number of branches in the decomposition."""
        return sum(len(self._branches[d]) for d in "NSEW")

    def all_rectangles(self, type: str = "*") -> Iterator[XY_Box]:
        """Iterator over all rectangles in the decomposition.
        type indicates the type of rectangles to be returned (T, N, S, E, W, *)"""
        assert type in ["T", "N", "S", "E", "W", "*"], (
            f"Invalid type {type} in all_rectangles"
        )

        if type in ["T"]:
            yield self._trunk
        elif type in ["N", "S", "E", "W"]:
            for b in self._branches[type]:
                yield b
        else:  # type == '*'
            yield self._trunk
            for d in "NSEW":
                for b in self._branches[d]:
                    yield b

    def area(self, type: str = "*") -> float:
        """Returns the area of the decomposition.
        If type is specified, only the area of that type of rectangles"""
        return sum(r.area for r in self.all_rectangles(type))

    @property
    def polygon(self) -> FPolygon:
        """Returns the polygon represented by the STROP decomposition."""
        return FPolygon(self.all_rectangles())

    def reduce(self) -> StropDecomposition:
        """Reduces the number of branches of the STROP by reducing the
        branches in one of the directions. It selectes the direction that
        maximizes the similarity with the original polygon.
        Returns a new StropDecomposition with the same original area."""
        assert self.num_branches > 0, "No branches to reduce"
        similarity = -1.0
        best_strop: Optional[StropDecomposition] = None
        for d in "NSEW":
            if len(self._branches[d]) > 0:
                new_strop = self.reduce_branches(d)
                if new_strop.similarity > similarity:
                    similarity = new_strop.similarity
                    best_strop = new_strop
        assert best_strop is not None
        return best_strop
    
    def reduce_branches(self, direction: str) -> StropDecomposition:
        """Reduces branches in the given direction by merging the
        closest pair of branches. In case only one branch remains, the branch
        is removed and the trunk is enlarged to keep the same area.
        Returns a new StropDecomposition with the same original area."""
        assert direction in "NSEW", f"Invalid direction {direction} in reduce_branches"
        num_branches = len(self._branches[direction])
        assert num_branches > 0, f"No branches in direction {direction} to reduce"
        
        new_strop = self.dup()
        if len(new_strop._branches[direction]) == 1:
            area = self.area(direction)
            tr = new_strop._trunk
            new_strop._branches[direction].clear()
            # Enlarge the trunk in the direction of the removed branches
            if direction == "N":
                tr.ymax += area / tr.width
            elif direction == "S":
                tr.ymin -= area / tr.width
            elif direction == "E":
                tr.xmax += area / tr.height
            else:  # direction == "W"
                tr.xmin -= area / tr.height
            new_strop._trunk = tr
            return new_strop

        # Find the smallest branch in the direction
        b1 = min(new_strop._branches[direction], key=lambda b: b.area)
        # Find the closest branch to b1 in the direction
        b2 = min(
            (b for b in new_strop._branches[direction] if b != b1),
            key=lambda b: b1.distance(b),
        )
        # Merge b1 and b2
        new_b = b1.iso_area_absorb(b2, direction)
        new_strop._branches[direction].remove(b1)
        new_strop._branches[direction].remove(b2)
        new_strop._branches[direction].add(new_b)
        
        # Previous version: merge the closest pair of branches until only one branch remains
        #b1, b2 = closest_boxes(new_strop._branches[direction])
        #new_b = b1.iso_area_merge(b2, direction)
        return new_strop



class FPolygon:
    """Class for polygons based on the rportion library.
    The polygon is represented as a union of rectangles with
    closed-open intervals."""

    _polygon: RPolygon  # The polygon represented in rportion

    def __init__(self, rectangles: Iterable[XY_Box] = list()):
        self._polygon = RPolygon()
        self._strop_decomposition = None
        for r in rectangles:
            self._polygon |= rclosedopen(
                float(r.xmin), float(r.xmax), float(r.ymin), float(r.ymax)
            )

    def dup(self) -> FPolygon:
        """Returns a deep copy of the Polygon."""
        c = FPolygon()
        c._polygon = copy.deepcopy(self._polygon)
        return c

    @property
    def area(self) -> float:
        """Returns the area of the polygon"""
        return sum(_RPoly2Box(r).area for r in self._polygon.rectangle_partitioning())

    def __or__(self, other: FPolygon) -> FPolygon:
        """Returns the union of polygons"""
        rec_copy = FPolygon()
        rec_copy._polygon = self._polygon | other._polygon
        return rec_copy

    def __and__(self, other: FPolygon) -> FPolygon:
        """Returns the intersection of polygons"""
        rec_copy = FPolygon()
        rec_copy._polygon = self._polygon & other._polygon
        return rec_copy

    def __sub__(self, other: FPolygon) -> FPolygon:
        """Returns the difference self-other"""
        rec_copy = FPolygon()
        rec_copy._polygon = self._polygon - other._polygon
        return rec_copy

    def __eq__(self, other: object) -> bool:
        """Checks equality of two polygons"""
        if not isinstance(other, FPolygon):
            return NotImplemented
        return self._polygon == other._polygon

    def __repr__(self) -> str:
        """Returns the representation of the polygon"""
        return repr(self._polygon)

    def jaccard_similarity(self, other: FPolygon) -> float:
        """Returns the Jaccard similarity between two polygons.
        The Jaccard similarity between two polygons P1 and P2 is a value
        in [0,1] defined as Area(P1&P2)/Area(P1|P2)."""
        return (self & other).area / (self | other).area

    def generate_all_strops(self) -> Iterator[StropDecomposition]:
        """Generates all possible STROP decompositions of the polygon."""
        for r in self._polygon.maximal_rectangles():
            trunk = FPolygon([_RPoly2Box(r)])
            strop = self.largest_strop(trunk)
            if strop.similarity == 1.0:
                yield strop

    def calculate_best_strop(self) -> Optional[StropDecomposition]:
        """Calculates the best STROP decomposition (the one with the fewer
        number of branches). If no STROP is found, None is returned"""
        # Calculate all strops generated by maximal rectangles
        all_strops = list(self.generate_all_strops())
        if not all_strops:
            return None
        # Select the best one: the one with the fewer number of branches.
        # In case of ties, the one with the largest trunk area
        return min(all_strops, key=lambda s: (s.num_branches, -s.trunk.area))

    def largest_strop(self, trunk: FPolygon) -> StropDecomposition:
        """Returns the largest strop included in self that has the
        associated trunk. The trunk is assumed to be a rectangle.
        """

        tr = _RPoly2Box(trunk._polygon)

        # Build the corners that must be subtracted
        ne = rclosedopen(tr.xmax, p.inf, tr.ymax, p.inf)
        nw = rclosedopen(-p.inf, tr.xmin, tr.ymax, p.inf)
        se = rclosedopen(tr.xmax, p.inf, -p.inf, tr.ymin)
        sw = rclosedopen(-p.inf, tr.xmin, -p.inf, tr.ymin)

        remainder = self._polygon - trunk._polygon - ne - nw - se - sw

        # The remainder covers all branches.

        # Classify the branches ('T', 'N', 'S', 'E', 'W')
        # Add the coordinate of the farthest edge to the trunk
        branches = list[tuple[RPolygon, str, float]]()
        for r in remainder.maximal_rectangles():
            b = _RPoly2Box(r)
            if not tr.touch(b):
                continue

            if b.xmax > tr.xmax:
                branches.append((r, "E", b.xmax))
            elif b.xmin < tr.xmin:
                branches.append((r, "W", -b.xmin))
            elif b.ymax > tr.ymax:
                branches.append((r, "N", b.ymax))
            elif b.ymin < tr.ymin:
                branches.append((r, "S", -b.ymin))
            else:
                raise ValueError("Unexpected branch")

        # Compute a disjoint set of branches.
        # Sort the branches by distance of the farthest edge to the trunk.
        # Then visit the branches and subtract the parts that overlap
        # with previous branches
        branches.sort(reverse=True, key=lambda r: r[2])
        prev_branches = RPolygon()
        strop = StropDecomposition(self, _RPoly2Box(trunk._polygon))

        for br in branches:
            new_b = br[0] - prev_branches
            if not new_b.empty:
                strop.add_rectangle(_RPoly2Box(new_b), br[1])
            prev_branches |= new_b

        return strop
