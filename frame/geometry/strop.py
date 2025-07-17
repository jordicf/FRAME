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
"""

from dataclasses import dataclass
from typing import Iterator

# Types to represent Boolean matrices
BoolRow = list[bool]
BoolMatrix = list[BoolRow]


@dataclass
class Interval:
    """Represents an interval of integer numbers.
    [-1, -1] represents the empty interval."""
    low: int
    high: int

    def empty(self) -> bool:
        """Returns whether the interval is empty"""
        return self.low == -1

    def intersection(self, other: 'Interval') -> 'Interval':
        """Returns the intersection of two intervals. If the intersection
        is empty, it returns the empty interval"""
        if self.empty() or other.empty():
            return EMPTY_INTERVAL
        i: Interval = Interval(max(self.low, other.low),
                               min(self.high, other.high))
        return i if i.low <= i.high else EMPTY_INTERVAL

    def length(self) -> int:
        """Returns the length of the interval"""
        if self.empty():
            return 0
        return self.high - self.low + 1


EMPTY_INTERVAL = Interval(-1, -1)


@dataclass
class Rectangle:
    """Represents a rectangle (interval of rows and interval of columns)"""
    rows: Interval
    columns: Interval

    def __hash__(self) -> int:
        """Hash function"""
        return hash((self.rows.low, self.rows.high,
                     self.columns.low, self.columns.high))

    def empty(self) -> bool:
        """Checks whether the rectangle is empty"""
        return self.rows.empty() or self.columns.empty()

    def area(self) -> int:
        """Returns the area (number of cells) of the rectangle"""
        return self.rows.length()*self.columns.length()


class Strop:
    """Class to represent orthogonal polygons"""
    _m: BoolMatrix  # matrix to represent the grid of the STrOP
    _nrows: int  # number of rows
    _ncols: int  # number of columns
    _height: list[float]  # heights of the rows
    _width: list[float]  # widths of the columns
    _instances: list['StropInstance']
    _valid: bool  # Is it a strop?

    def __init__(self, str_matrix: str, height: list[float] = list(),
                 width: list[float] = list()):
        """Constructor from a Boolean matrix represented as a string of 0's
        and 1's. Each row is separated by a whitespace.
        height ahd width represent the heights and the widths of the rows and
        columns, respectively.
        If the lists are empty, unit sizes are assumed"""
        lst = str_matrix.split()  # Split as a list of strings
        self._nrows = len(lst)
        assert self._nrows > 0, "Creating an empty STrOP"
        self._ncols = len(lst[0])
        assert all(
            len(x) == self._ncols for x in lst), \
            "Illegal STrOP: rows with different size"
        assert all(lst[i][j] in {'0', '1'}
                   for i in range(self._nrows) for j in range(self._ncols)), \
            "Non-binary elements in Boolean matrix"
        self._m = [[lst[i][j] == '1' for j in range(self._ncols)]
                   for i in range(self._nrows)]
        # Heights and widths
        assert len(height) == 0 or len(
            height) == self._nrows, "Wrong number of rows in height"
        assert len(width) == 0 or len(
            width) == self._ncols, "Wrong number of columns in width"
        self._height = height[:] if len(height) > 0 else [1]*self._nrows
        self._width = width[:] if len(width) > 0 else [1]*self._ncols

        # Generate OrthoTrees
        self._instances = list()
        for trunk in self._get_potential_trunks():
            ot = StropInstance(self, trunk)
            if ot.valid():
                self._instances.append(ot)

    @property
    def num_rows(self) -> int:
        """Returns the number of rows of the grid"""
        return self._nrows

    @property
    def num_columns(self) -> int:
        """Returns the number of columns of the grid"""
        return self._ncols

    @property
    def is_strop(self) -> bool:
        """Indicates whether the polygon is a valid strop"""
        return len(self._instances) > 0

    @property
    def matrix(self) -> BoolMatrix:
        """Returns the boolean matrix of the grid"""
        return self._m

    @property
    def get_width(self) -> list[float]:
        """Returns the list of widths of the grid"""
        return self._width

    def instances(self) -> Iterator['StropInstance']:
        """Generates a list of orthogonal trees (trunk+branches)"""
        for tree in self._instances:
            yield tree

    def _get_potential_trunks(self) -> set[Rectangle]:
        """Returns a set of rectangles that could be potentially
        trunks of the polygon"""
        Mt = [[self._m[j][i]
               for j in range(self._nrows)] for i in range(self._ncols)]
        return {
            t for t in Strop._get_trunks_matrix(self._m).intersection(
                {Rectangle(r.columns, r.rows)
                 for r in Strop._get_trunks_matrix(Mt)})
            if self._empty_corners(t)
        }

    def _empty_corners(self, R: Rectangle) -> bool:
        """It indicates if the NE, NW, SE, SW corners of the rectangle are
        empty in the matrix"""

        return not (
            any(self._m[i][j] for i in range(R.rows.low)
                for j in range(R.columns.low))
            or
            any(self._m[i][j] for i in range(R.rows.low)
                for j in range(R.columns.high+1, self._ncols))
            or
            any(self._m[i][j] for i in range(R.rows.high+1, self._nrows)
                for j in range(R.columns.low))
            or
            any(self._m[i][j] for i in range(R.rows.high+1, self._nrows)
                for j in range(R.columns.high+1, self._ncols))
        )

    @staticmethod
    def _get_trunks_matrix(M: BoolMatrix) -> set[Rectangle]:
        """Returns a list of rectangles that could potentially be the trunk
        of the polygon"""
        nrows = len(M)
        rect: list[list[Interval]] = \
            [[EMPTY_INTERVAL]*nrows for _ in range(nrows)]
        # Fill-up diagonals with the longes interval of columns
        for i in range(nrows):
            rect[i][i] = Strop._row_interval(M[i])

        # Now fill up the upper triangle
        for column in range(1, nrows):
            for row in range(column-1, -1, -1):
                rect[row][column] = \
                    rect[row + 1][column].intersection(rect[row][column-1])

        # Remove the non-prime rectangles by rows
        for row in range(nrows-1):
            for column in range(row, nrows-1):
                if rect[row][column] == rect[row][column+1]:
                    rect[row][column] = EMPTY_INTERVAL

        # Remove the non-prime rectangles by columns
        for column in range(1, nrows):
            for row in range(column, 0, -1):
                if rect[row][column] == rect[row-1][column]:
                    rect[row][column] = EMPTY_INTERVAL
        return {Rectangle(Interval(row, column), rect[row][column])
                for row in range(nrows)
                for column in range(row, nrows)
                if rect[row][column] != EMPTY_INTERVAL}

    @staticmethod
    def _row_interval(R: BoolRow) -> Interval:
        """Finds the longest interval in the boolean row. If there is more
        than one connected interval, the empty interval is returned.
        A trunk cannot cover a row with more than one interval."""
        # Find the first True
        try:
            first_true = R.index(True)
        except ValueError:
            return EMPTY_INTERVAL  # No True: empty row

        # Find the first False after the first True
        try:
            first_false = R.index(False, first_true + 1)
        except ValueError:
            return Interval(first_true, len(R)-1)  # segment at the tail

        # Find a new True. If it exists, bad row for a trunk
        try:
            R.index(True, first_false + 1)
            return EMPTY_INTERVAL  # Found, not a good row
        except ValueError:
            return Interval(first_true, first_false-1)  # Not found, good row


class StropInstance:
    """Class to represent an instance of a STrOP"""
    _poly: Strop  # Representation of the orthogonal polygon
    _trunk: Rectangle  # Trunk
    _north: list[Rectangle]  # North branches
    _south: list[Rectangle]  # South branches
    _east: list[Rectangle]  # East branches
    _west: list[Rectangle]  # West branches
    _num_cells: int  # number of cells of the OrthoTree
    _valid: bool  # Is the OrthoTree a Strop?

    def __init__(self, p: Strop, trunk: Rectangle):
        self._poly = p
        self._trunk = trunk
        m = p.matrix
        self._num_cells = sum(1 for i in range(p.num_rows)
                              for j in range(p.num_columns) if m[i][j])
        # Generate histograms for the borders of the trunk
        h_north: list[int] = [0]*p.num_columns
        h_south: list[int] = [0]*p.num_columns
        h_east: list[int] = [0]*p.num_rows
        h_west: list[int] = [0]*p.num_rows

        # In total we will accumulate the area of trunk and branches
        total = trunk.area()

        # North and South histograms
        for c in range(trunk.columns.low, trunk.columns.high+1):
            for r in range(trunk.rows.low-1, -1, -1):
                if not m[r][c]:
                    break
                h_north[c] += 1
                total += 1
            for r in range(trunk.rows.high+1, p.num_rows):
                if not m[r][c]:
                    break
                h_south[c] += 1
                total += 1

        # East and West histograms
        for r in range(trunk.rows.low, trunk.rows.high+1):
            for c in range(trunk.columns.low-1, -1, -1):
                if not m[r][c]:
                    break
                h_west[r] += 1
                total += 1
            for c in range(trunk.columns.high+1, p.num_columns):
                if not m[r][c]:
                    break
                h_east[r] += 1
                total += 1

        # Check that area of trunk and branches is the total area
        self._valid = self._num_cells == total
        if not self.valid():
            return

        self._north, self._south, self._east, self._west = \
            list(), list(), list(), list()
        # Generate the north branches visiting the north histogram
        init_c, v = trunk.columns.low, h_north[trunk.columns.low]
        for c in range(trunk.columns.low + 1, trunk.columns.high+1):
            if h_north[c] != v:
                if v != 0:
                    self._north.append(Rectangle(
                        Interval(trunk.rows.low-v, trunk.rows.low-1),
                        Interval(init_c, c-1)))
                init_c = c
                v = h_north[c]
        if v != 0:  # Last rectangle
            self._north.append(Rectangle(
                Interval(trunk.rows.low-v, trunk.rows.low-1),
                Interval(init_c, trunk.columns.high)))

        # Generate the south branches
        init_c, v = trunk.columns.low, h_south[trunk.columns.low]
        for c in range(trunk.columns.low + 1, trunk.columns.high+1):
            if h_south[c] != v:
                if v != 0:
                    self._south.append(Rectangle(
                        Interval(trunk.rows.high+1, trunk.rows.high+v),
                        Interval(init_c, c-1)))
                init_c = c
                v = h_south[c]
        if v != 0:  # Last rectangle
            self._south.append(Rectangle(
                Interval(trunk.rows.high+1, trunk.rows.high+v),
                Interval(init_c, trunk.columns.high)))

        # Generate the west branches
        init_r, v = trunk.rows.low, h_west[trunk.rows.low]
        for r in range(trunk.rows.low + 1, trunk.rows.high+1):
            if h_west[r] != v:
                if v != 0:
                    self._west.append(Rectangle(
                        Interval(init_r, r-1),
                        Interval(trunk.columns.low-v, trunk.columns.low-1)))
                init_r = r
                v = h_west[r]
        if v != 0:  # Last rectangle
            self._west.append(Rectangle(
                Interval(init_r, trunk.rows.high),
                Interval(trunk.columns.low-v, trunk.columns.low-1)))

        # Generate the east branches
        init_r, v = trunk.rows.low, h_east[trunk.rows.low]
        for r in range(trunk.rows.low + 1, trunk.rows.high+1):
            if h_east[r] != v:
                if v != 0:
                    self._east.append(Rectangle(
                        Interval(init_r, r-1),
                        Interval(trunk.columns.high+1, trunk.columns.high+v)))
                init_r = r
                v = h_east[r]
        if v != 0:  # Last rectangle
            self._east.append(Rectangle(
                Interval(init_r, trunk.rows.high),
                Interval(trunk.columns.high+1, trunk.columns.high+v)))

    def valid(self) -> bool:
        """Reports whether it is a valid STROP"""
        return self._valid

    def trunk(self) -> Rectangle:
        """Returns the trunk of the tree"""
        return self._trunk

    def rectangles(self, which: str = '') -> Iterator[Rectangle]:
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
        assert all(x in {'N', 'S', 'W', 'E', 'T', 'B'} for x in which), \
            "Unknown type of rectangles"

        if not which or 'T' in which:
            yield self.trunk()

        for side, lst in [('N', self._north), ('S', self._south),
                          ('E', self._east), ('W', self._west)]:
            if not which or side in which or 'B' in which:
                for r in lst:
                    yield r

    def __str__(self) -> str:
        """Returns a string representing the STROP.
        The string represents a grid in which 0 represents the trunk and
        1, 2, ... represent the branches."""
        grid: list[list[str]] = [[' ']*self._poly.num_columns
                                 for _ in range(self._poly.num_rows)]
        idx = 0
        for r in self.rectangles():  # all rectangles
            StropInstance._str_rectangle(grid, r, idx)
            idx += 1

        # Create the string representing the matrix
        return '\n'.join([''.join(row).rstrip() for row in grid])

    @staticmethod
    def _str_rectangle(grid: list[list[str]], rect: Rectangle,
                       idx: int) -> None:
        """Defines the cells of the grid with the rectangle rect and
        index idx"""
        for r in range(rect.rows.low, rect.rows.high+1):
            for c in range(rect.columns.low, rect.columns.high+1):
                grid[r][c] = f"{idx:x}"
