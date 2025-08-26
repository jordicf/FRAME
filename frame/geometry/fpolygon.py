# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"This module implements a polygon class based on the rportion library."

from __future__ import annotations
import portion as p
from rportion import RPolygon, rclosedopen
from copy import copy
from typing import Iterable, Optional

# Some auxiliary types
# Tuple to represent a rectangle in the interval [xmin, xmax, ymin, ymax]
XY_Box = tuple[float, float, float, float]

# Tuple to represent a rectangle and its location in a STROP
# The string can be 'T' (trunk), 'N', 'S', 'E', 'W'
StropRectangle = tuple[XY_Box, str]
StropDecomposition = list[StropRectangle]

# Some methods to extend the functionality of rportion


def _RPolygon2Box(r: RPolygon) -> XY_Box:
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
    return (xmin, xmax, ymin, ymax)


def _touch(p1: RPolygon, p2: RPolygon) -> bool:
    """Returns True if the two polygons touch each other (i.e., they have
    at least one point in common)."""
    p1_xmin, p1_xmax, p1_ymin, p1_ymax = _RPolygon2Box(p1)
    p2_xmin, p2_xmax, p2_ymin, p2_ymax = _RPolygon2Box(p2)
    return (
        p1_xmin <= p2_xmax
        and p1_xmax >= p2_xmin
        and p1_ymin <= p2_ymax
        and p1_ymax >= p2_ymin
    )


class FPolygon:
    """Class for polygons based on the rportion library.
    The polygon is represented as a union of rectangles with
    closed-open intervals."""

    _polygon: RPolygon  # The polygon represented in rportion
    _area: float  # Area of the polygon
    _nrectangles: int  # Number of rectangles after decomposition
    # STROP decomposition. It is represented as a list of rectangles with
    # the associated location in the STROP ('T', 'N', 'S', 'E', 'W')
    _strop_decomposition: Optional[StropDecomposition]

    def __init__(self, rectangles: Iterable[XY_Box] = list()):
        self._polygon = RPolygon()
        self._area = -1
        self._nrectangles = -1
        self._strop_decomposition = None
        for r in rectangles:
            self._polygon |= rclosedopen(
                float(r[0]), float(r[1]), float(r[2]), float(r[3])
            )

    @property
    def area(self) -> float:
        """Returns the area of the polygon"""
        if self._area < 0:  # If undefined, compute area
            self._area = 0
            for r in self._polygon.rectangle_partitioning():
                xmin, xmax, ymin, ymax = _RPolygon2Box(r)
                self._area += (xmax - xmin) * (ymax - ymin)
        return self._area

    @property
    def num_rectangles(self) -> int:
        """Defines the number of rectangles of the polygon"""
        if self._nrectangles < 0:
            self._nrectangles = sum(1 for _ in self._polygon.rectangle_partitioning())
        return self._nrectangles

    @property
    def strop_decomposition(self) -> Optional[StropDecomposition]:
        """Returns the STROP decomposition of the polygon, if it has been
        computed."""
        return self._strop_decomposition

    @property
    def num_strop_branches(self) -> int:
        """Returns the number of branches in the STROP decomposition."""
        if self._strop_decomposition is None:
            return 0
        return len(self._strop_decomposition) - 1

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

    def __copy__(self) -> FPolygon:
        """Returns a deep copy of the Polygon."""
        c = FPolygon()
        c._area = self._area
        c._nrectangles = self._nrectangles
        c._polygon = copy(self._polygon)
        return c

    def __repr__(self) -> str:
        """Returns the representation of the polygon"""
        if self._strop_decomposition is None:
            return repr(self._polygon)
        return (
            "STROP("
            + ", ".join([f"{r[0]}:{r[1]}" for r in self._strop_decomposition])
            + ")"
        )

    def copy(self) -> FPolygon:
        """Returns a deep copy of the Polygon."""
        return copy(self)

    def jaccard_similarity(self, other: FPolygon) -> float:
        """Returns the Jaccard similarity between two polygons.
        The Jaccard similarity between two polygons P1 and P2 is a value
        in [0,1] defined as Area(P1&P2)/Area(P1|P2)."""
        return (self & other).area / (self | other).area

    def calculate_best_strop(self) -> bool:
        """Calculates the best STROP decomposition (the one with the fewer
        number of branches). Returns True if the polygon is a STROP"""
        # Calculate all strops generated by maximal rectangles
        best: Optional[FPolygon] = None
        best_nbranches = -1
        best_area = 0.0
        for r in self._polygon.maximal_rectangles():
            trunk = FPolygon([_RPolygon2Box(r)])
            strop = self.largest_strop(trunk)
            if strop.strop_decomposition is None or strop != self:
                continue

            nbranches = strop.num_strop_branches
            if (
                best is None
                or (nbranches < best_nbranches)
                or (nbranches == best_nbranches and trunk.area > best_area)
            ):
                best = strop
                best_nbranches = nbranches
                best_area = strop.area

        if best is None:
            return False

        self._strop_decomposition = best.strop_decomposition
        return True

    def largest_strop(self, trunk: FPolygon) -> FPolygon:
        """Returns the largest strop included in self that has the
        associated trunk. The trunk is assumed to be a rectangle.
        The strop includes it strop decomposition."""

        xmin, xmax, ymin, ymax = _RPolygon2Box(trunk._polygon)

        # Build the corners that must be subtracted
        ne = rclosedopen(xmax, p.inf, ymax, p.inf)
        nw = rclosedopen(-p.inf, xmin, ymax, p.inf)
        se = rclosedopen(xmax, p.inf, -p.inf, ymin)
        sw = rclosedopen(-p.inf, xmin, -p.inf, ymin)

        remainder = self._polygon - trunk._polygon - ne - nw - se - sw

        # The remainder are all branches.

        # Classify the branches ('T', 'N', 'S', 'E', 'W')
        # Add the coordinate of the farthest edge to the trunk
        branches = list[tuple[RPolygon, str, float]]()
        for r in remainder.maximal_rectangles():
            if not _touch(trunk._polygon, r):
                continue
            rxmin, rxmax, rymin, rymax = _RPolygon2Box(r)
            if rxmax > xmax:
                branches.append((r, "E", rxmax))
            elif rxmin < xmin:
                branches.append((r, "W", -rxmin))
            elif rymax > ymax:
                branches.append((r, "N", rymax))
            elif rymin < ymin:
                branches.append((r, "S", -rymin))
            else:
                raise ValueError("Unexpected branch")

        # Compute a disjoint set of branches.
        # Sort the branches by distance of the farthest edge to the trunk.
        # Then visit the branches and subtract the parts that overlap
        # with previous branches
        branches.sort(reverse=True, key=lambda r: r[2])
        prev_branches = RPolygon()
        strop: StropDecomposition = [(_RPolygon2Box(trunk._polygon), "T")]
        for b in branches:
            new_b = b[0] - prev_branches
            if not new_b.empty:
                strop.append((_RPolygon2Box(new_b), b[1]))
            prev_branches |= new_b

        poly = FPolygon()
        poly._polygon = trunk._polygon | prev_branches
        poly._strop_decomposition = strop
        return poly
