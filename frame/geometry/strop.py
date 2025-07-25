# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
This module manipulates STROPs. Originally this acronym was for
Single-Trunk Orthogonal Polygons but it can also be read as
Star Orthogonal Polygons.

A STROP is an orthogonal polygon that can be represented by a set of
connected disjoint rectangles. One of the rectangles is the trunk.
The other rectangles are branches in one of the sides NSEW.

The property of a STROP is that branches are fully adjacent to the trunk.
This is a STROP:
       22
    11 22
   0000000
 3300000004
 3300000004
   0000000

The trunk is the rectangle represented by 0's. The STROP has two branches
at the north (1 and 2), one branch at the west (3) and another at the east (4).

This is not a STROP:

 XX
  XX
   X

Any attempt to identify a trunk ends up by finding branches that are not
fully adjacent to the trunk.

STROPs are represented as a set of rectangles in a Boolean matrix (grid).
This grid may be the Hanan grid of a floorplan.
The upper-left corner has row=0 and column=0.

Each rectangle is represented as an interval of rows and columns in the grid.

The class Strop is initialized with an occupancy matrix. This matrix
is assumed to represent the occupancy of a module in a Hanan grid.
Each cell contains the occupancy of the module. For example, if three modules
occupy the same Hanan cell, the occupancy for each one will be 1/3.

Internally, Strop contains a Boolean matrix that indicates all cells occupied
by a module (occupancy > 0).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Final
import numpy as np
import numpy.typing as npt

# Types to represent Boolean matrices and occupancy matrices =
BoolRow = npt.NDArray[np.bool]
BoolMatrix = npt.NDArray[np.bool]
OccMatrix = npt.NDArray[np.float64]


@dataclass
class Interval:
    """Represents an interval of integer numbers.
    [-1, -1] represents the empty interval."""

    low: int
    high: int

    def is_empty(self) -> bool:
        """Returns whether the interval is empty"""
        return self.low < 0 or self.high < 0

    def intersection(self, other: "Interval") -> "Interval":
        """Returns the intersection of two intervals. If the intersection
        is empty, it returns the empty interval"""
        if self.is_empty() or other.is_empty():
            return EMPTY_INTERVAL
        i: Interval = Interval(max(self.low, other.low), min(self.high, other.high))
        return i if i.low <= i.high else EMPTY_INTERVAL

    def length(self) -> int:
        """Returns the length of the interval"""
        if self.is_empty():
            return 0
        return self.high - self.low + 1


EMPTY_INTERVAL: Final = Interval(-1, -1)


@dataclass
class Rectangle:
    """Represents a rectangle (interval of rows and interval of columns)"""

    rows: Interval
    columns: Interval

    def __hash__(self) -> int:
        """Hash function"""
        return hash(
            (self.rows.low, self.rows.high, self.columns.low, self.columns.high)
        )

    def empty(self) -> bool:
        """Checks whether the rectangle is empty"""
        return self.rows.is_empty() or self.columns.is_empty()

    def area(self) -> int:
        """Returns the area (number of cells) of the rectangle"""
        return self.rows.length() * self.columns.length()

    def shift(self, row_off: int, col_off: int) -> None:
        """Relocates the rectangle by adding the row and column offsets
        to the row and column intervals."""
        self.rows.low += row_off
        self.rows.high += row_off
        self.columns.low += col_off
        self.columns.high += col_off


class Strop:
    """Class to represent orthogonal polygons"""

    _occ: OccMatrix  # Occupancy matrix (values in [0,1])
    _bm: BoolMatrix  # Boolean matrix associated to _occ (True if _occ > 0)
    _height: list[float]  # heights of the rows
    _width: list[float]  # widths of the columns
    _first_row: int  # first_row after trimming zeros
    _last_row: int  # last_row after trimming zeros
    _first_col: int  # first column after trimming zeros
    _last_col: int  # last column after trimming zeros
    _instances: list["StropInstance"]
    _valid: bool  # Is it a strop?

    def __init__(
        self,
        occ_matrix: OccMatrix,
        height: list[float] = list(),
        width: list[float] = list(),
    ):
        """Constructor from an occupancy matrix with values in [0,1].
        height ahd width represent the heights and the widths of the rows and
        columns, respectively. If the lists are empty, unit sizes are assumed"""

        # Let us first trim the matrix and remove
        # peripheral rows and columnswith zeros

        nz = np.nonzero(occ_matrix)
        self._first_row, self._last_row = min(nz[0]), max(nz[0])
        self._first_col, self._last_col = min(nz[1]), max(nz[1])
        self._occ = np.array(
            occ_matrix[
                self._first_row : self._last_row + 1,
                self._first_col : self._last_col + 1,
            ],
            dtype=np.float64,
        )

        self._bm = np.array(self._occ, dtype=bool)

        # Heights and widths
        shape = occ_matrix.shape
        assert len(height) == 0 or len(height) == shape[0], (
            "Wrong number of rows in height"
        )
        assert len(width) == 0 or len(width) == shape[1], (
            "Wrong number of columns in width"
        )
        self._height = (
            height[self._first_row : self._last_row + 1]
            if len(height) > 0
            else [1] * self.num_rows
        )

        self._width = (
            width[self._first_col : self._last_col + 1]
            if len(width) > 0
            else [1] * self.num_columns
        )

        # Generate STROPs
        self._instances = list()
        for trunk in self._get_potential_trunks():
            ot = StropInstance(self, trunk)
            if ot.valid():
                self._instances.append(ot)

    @property
    def num_rows(self) -> int:
        """Returns the number of rows of the grid"""
        return len(self._bm)

    @property
    def num_columns(self) -> int:
        """Returns the number of columns of the grid"""
        return self._bm.shape[1]

    @property
    def is_strop(self) -> bool:
        """Indicates whether the polygon is a valid strop"""
        return len(self._instances) > 0

    @property
    def matrix(self) -> BoolMatrix:
        """Returns the boolean matrix of the grid"""
        return self._bm

    @property
    def get_width(self) -> list[float]:
        """Returns the list of widths of the grid"""
        return self._width

    def instances(self) -> Iterator["StropInstance"]:
        """Generates a list of orthogonal trees (trunk+branches)"""
        for tree in self._instances:
            yield tree

    def _get_potential_trunks(self) -> set[Rectangle]:
        """Returns a set of rectangles that could be potentially
        trunks of the polygon"""
        Mt = np.transpose(self._bm)
        return {
            t
            for t in Strop._get_trunks_matrix(self._bm).intersection(
                {Rectangle(r.columns, r.rows) for r in Strop._get_trunks_matrix(Mt)}
            )
            if self._empty_corners(t)
        }

    def _empty_corners(self, R: Rectangle) -> bool:
        """It indicates if the NE, NW, SE, SW corners of the rectangle are
        empty in the matrix"""

        return not (
            any(self._bm[i][j] for i in range(R.rows.low) for j in range(R.columns.low))
            or any(
                self._bm[i][j]
                for i in range(R.rows.low)
                for j in range(R.columns.high + 1, self.num_columns)
            )
            or any(
                self._bm[i][j]
                for i in range(R.rows.high + 1, self.num_rows)
                for j in range(R.columns.low)
            )
            or any(
                self._bm[i][j]
                for i in range(R.rows.high + 1, self.num_rows)
                for j in range(R.columns.high + 1, self.num_columns)
            )
        )

    @staticmethod
    def _get_trunks_matrix(M: BoolMatrix) -> set[Rectangle]:
        """Returns a list of rectangles that could potentially be the trunk
        of the polygon"""
        nrows = len(M)
        # We build a square matrix rect (nrows x nrows).
        # rect[i][j] represents the largest interval of columns in M
        # for a rectangle between rows i and j (i <= j)
        rect: list[list[Interval]] = [[EMPTY_INTERVAL] * nrows for _ in range(nrows)]
        # Fill-up diagonals with the longest interval of columns at row i
        for i in range(nrows):
            rect[i][i] = Strop._row_interval(M[i])

        # Now fill up the upper triangle
        for last_row in range(1, nrows):
            for row in range(last_row - 1, -1, -1):
                rect[row][last_row] = rect[row + 1][last_row].intersection(
                    rect[row][row]
                )

        # Remove the non-prime rectangles by rows
        for row in range(nrows - 1):
            for last_row in range(row, nrows - 1):
                if rect[row][last_row] == rect[row][last_row + 1]:
                    rect[row][last_row] = EMPTY_INTERVAL

        # Remove the non-prime rectangles by columns
        for last_row in range(1, nrows):
            for row in range(last_row, 0, -1):
                if rect[row][last_row] == rect[row - 1][last_row]:
                    rect[row][last_row] = EMPTY_INTERVAL

        # Return the non-empty intervals
        return {
            Rectangle(Interval(row, last_row), rect[row][last_row])
            for row in range(nrows)
            for last_row in range(row, nrows)
            if rect[row][last_row] != EMPTY_INTERVAL
        }

    @staticmethod
    def _row_interval(R: BoolRow) -> Interval:
        """Finds the longest interval in the boolean row. If there is more
        than one connected interval, the empty interval is returned.
        A trunk cannot cover a row with more than one interval."""
        all_trues = np.where(R)[0]  # Indices of the True positions
        num_trues = len(all_trues)
        if num_trues == 0:
            return EMPTY_INTERVAL  # No True: empty row

        # Check that all True's are contiguous
        if all_trues[-1] - all_trues[0] + 1 == num_trues:
            return Interval(all_trues[0], all_trues[-1])

        return EMPTY_INTERVAL


class StropInstance:
    """Class to represent an instance of a STROP"""

    _poly: Strop  # Representation of the orthogonal polygon
    _row_off: int  # Row offset in the Hanan grid
    _col_off: int  # Column offset in the Hanan grid
    _trunk: Rectangle  # Trunk
    _north: list[Rectangle]  # North branches
    _south: list[Rectangle]  # South branches
    _east: list[Rectangle]  # East branches
    _west: list[Rectangle]  # West branches
    _num_cells: int  # number of cells of the OrthoTree
    _valid: bool  # Is the OrthoTree a Strop?

    def __init__(self, p: Strop, trunk: Rectangle, row_off: int = 0, col_off: int = 0):
        """Constructor of a STROP instance. The original polygon is
        represented in a matrix inside a Hana grid. row_off and col_off are
        the offsets that must be applied to obtain the actual row and column
        indices in the Hanan grid.
        :param p: the original polygon
        :param trunk: the trunk around which the STROP must be built
        :param row_off: index of the first row of the Hanan grid"""
        self._poly = p
        self._row_off, self._col_off = row_off, col_off
        self._trunk = trunk
        m = p.matrix
        self._num_cells = sum(
            1 for i in range(p.num_rows) for j in range(p.num_columns) if m[i][j]
        )
        # Generate histograms for the borders of the trunk
        h_north: list[int] = [0] * p.num_columns
        h_south: list[int] = [0] * p.num_columns
        h_east: list[int] = [0] * p.num_rows
        h_west: list[int] = [0] * p.num_rows

        # In total we will accumulate the area of trunk and branches
        total = trunk.area()

        # North and South histograms
        for c in range(trunk.columns.low, trunk.columns.high + 1):
            for r in range(trunk.rows.low - 1, -1, -1):
                if not m[r][c]:
                    break
                h_north[c] += 1
                total += 1
            for r in range(trunk.rows.high + 1, p.num_rows):
                if not m[r][c]:
                    break
                h_south[c] += 1
                total += 1

        # East and West histograms
        for r in range(trunk.rows.low, trunk.rows.high + 1):
            for c in range(trunk.columns.low - 1, -1, -1):
                if not m[r][c]:
                    break
                h_west[r] += 1
                total += 1
            for c in range(trunk.columns.high + 1, p.num_columns):
                if not m[r][c]:
                    break
                h_east[r] += 1
                total += 1

        # Check that area of trunk and branches is the total area
        self._valid = self._num_cells == total
        if not self.valid():
            return

        self._north, self._south, self._east, self._west = (
            list(),
            list(),
            list(),
            list(),
        )
        # Generate the north branches visiting the north histogram
        init_c, v = trunk.columns.low, h_north[trunk.columns.low]
        for c in range(trunk.columns.low + 1, trunk.columns.high + 1):
            if h_north[c] != v:
                if v != 0:
                    self._north.append(
                        Rectangle(
                            Interval(trunk.rows.low - v, trunk.rows.low - 1),
                            Interval(init_c, c - 1),
                        )
                    )
                init_c = c
                v = h_north[c]
        if v != 0:  # Last rectangle
            self._north.append(
                Rectangle(
                    Interval(trunk.rows.low - v, trunk.rows.low - 1),
                    Interval(init_c, trunk.columns.high),
                )
            )

        # Generate the south branches
        init_c, v = trunk.columns.low, h_south[trunk.columns.low]
        for c in range(trunk.columns.low + 1, trunk.columns.high + 1):
            if h_south[c] != v:
                if v != 0:
                    self._south.append(
                        Rectangle(
                            Interval(trunk.rows.high + 1, trunk.rows.high + v),
                            Interval(init_c, c - 1),
                        )
                    )
                init_c = c
                v = h_south[c]
        if v != 0:  # Last rectangle
            self._south.append(
                Rectangle(
                    Interval(trunk.rows.high + 1, trunk.rows.high + v),
                    Interval(init_c, trunk.columns.high),
                )
            )

        # Generate the west branches
        init_r, v = trunk.rows.low, h_west[trunk.rows.low]
        for r in range(trunk.rows.low + 1, trunk.rows.high + 1):
            if h_west[r] != v:
                if v != 0:
                    self._west.append(
                        Rectangle(
                            Interval(init_r, r - 1),
                            Interval(trunk.columns.low - v, trunk.columns.low - 1),
                        )
                    )
                init_r = r
                v = h_west[r]
        if v != 0:  # Last rectangle
            self._west.append(
                Rectangle(
                    Interval(init_r, trunk.rows.high),
                    Interval(trunk.columns.low - v, trunk.columns.low - 1),
                )
            )

        # Generate the east branches
        init_r, v = trunk.rows.low, h_east[trunk.rows.low]
        for r in range(trunk.rows.low + 1, trunk.rows.high + 1):
            if h_east[r] != v:
                if v != 0:
                    self._east.append(
                        Rectangle(
                            Interval(init_r, r - 1),
                            Interval(trunk.columns.high + 1, trunk.columns.high + v),
                        )
                    )
                init_r = r
                v = h_east[r]
        if v != 0:  # Last rectangle
            self._east.append(
                Rectangle(
                    Interval(init_r, trunk.rows.high),
                    Interval(trunk.columns.high + 1, trunk.columns.high + v),
                )
            )

    def valid(self) -> bool:
        """Reports whether it is a valid STROP"""
        return self._valid

    def trunk(self) -> Rectangle:
        """Returns the trunk of the tree"""
        return self._trunk

    def rectangles(self, which: str = "") -> Iterator[Rectangle]:
        """Returns the rectangles of the STROP instance depending on the
        value of which.
        '' -> all rectangles (default)
        'T' -> the trunk
        'B' -> all branches
        'N', 'S', 'E', 'W'-> north, south, east, west branches
        any combination of the previous characters in the string which,
        e.g. 'NST'
        """
        assert self.valid(), "Invalid STROP"
        assert all(x in {"N", "S", "W", "E", "T", "B"} for x in which), (
            "Unknown type of rectangles"
        )

        if not which or "T" in which:
            yield self.trunk()

        for side, lst in [
            ("N", self._north),
            ("S", self._south),
            ("E", self._east),
            ("W", self._west),
        ]:
            if not which or side in which or "B" in which:
                for r in lst:
                    yield r

    def shift(self) -> None:
        """Relocates all rectangles by adding the row and column offsets
        to the row and column intervals."""
        self._trunk.shift(self._row_off, self._col_off)
        for side in [self._north, self._south, self._east, self._west]:
            for r in side:
                r.shift(self._row_off, self._col_off)

    def __str__(self) -> str:
        """Returns a string representing the STROP.
        The string represents a grid in which 0 represents the trunk and
        1, 2, ... represent the branches."""
        grid: list[list[str]] = [
            [" "] * self._poly.num_columns for _ in range(self._poly.num_rows)
        ]
        idx = 0
        for r in self.rectangles():  # all rectangles
            StropInstance._str_rectangle(grid, r, idx, self._row_off, self._col_off)
            idx += 1

        # Create the string representing the matrix
        return "\n".join(["".join(row).rstrip() for row in grid])

    @staticmethod
    def _str_rectangle(
        grid: list[list[str]],
        rect: Rectangle,
        idx: int,
        row_off: int = 0,
        col_off: int = 0,
    ) -> None:
        """Defines the cells of the grid with the rectangle rect and
        index idx. The coordinates of the rectangles are shifter by
        the row and column offsets"""
        for r in range(rect.rows.low, rect.rows.high + 1):
            for c in range(rect.columns.low, rect.columns.high + 1):
                grid[r - row_off][c - col_off] = f"{idx:x}"
