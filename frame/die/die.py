from collections import deque
from itertools import combinations
from typing import Set, Deque
from dataclasses import dataclass

from .yaml_parse_die import parse_yaml_die
from frame.geometry.geometry import Shape, Rectangle, Point, gather_boundaries, split_rectangles
from frame.netlist.netlist import Netlist
from frame.utils.keywords import KW_CENTER, KW_SHAPE, KW_REGION, KW_GROUND, KW_BLOCKAGE
from frame.utils.utils import TextIO_String


@dataclass
class GroundRegion:
    """Representation of a ground region in a cell matrix"""
    rmin: int  # min row
    rmax: int  # max row
    cmin: int  # min column
    cmax: int  # max column
    area: float  # area of the cell
    ratio: float  # aspect ratio

    def __str__(self) -> str:
        return f'<rows=({self.rmin}-{self.rmax}), cols=({self.cmin}-{self.cmax}), area={self.area}, ratio={self.ratio}>'

    def __hash__(self) -> int:
        return hash(37 * self.rmin + 13 * self.rmax + 7 * self.cmin + 23 * self.cmax)


class Die:
    """
    Class to represent the die (ground and tagged rectangles)
    """
    _netlist: Netlist | None  # Netlist associated to the die
    _die: Rectangle  # Bounding Box of the die
    _specialized_regions: list[Rectangle]  # List of non-ground regions
    _ground_regions: list[Rectangle]  # List of ground regions (not covered by fixed rectangles)
    _blockages: list[Rectangle]  # List of blockages
    _fixed: list[Rectangle]  # List of fixed rectangles (obtained from a netlist)
    _epsilon: float  # Precision when dealing with coordinates
    _x: list[float]  # List of x coordinates of potential rectangles
    _y: list[float]  # List of y coordinates of potential rectangles
    _cells: list[list[bool]]  # Matrix of rectangles (True occupied, False available)

    def __init__(self, stream: TextIO_String, netlist: Netlist | None = None):
        """
        Constructor of a die from a file or from a string of text
        :param stream: name of the YAML file (str) or handle to the file
        :param netlist: the netlist associated to the die (necessary for fixed modules)
        """
        regions: list[Rectangle]
        self._netlist = netlist
        self._die, regions = parse_yaml_die(stream)
        self._epsilon = min(self.width, self.height) * 10e-12

        # Selectec blockages from the other regions
        self._specialized_regions, self._blockages = [], []
        for r in regions:
            self._blockages.append(r) if r.region == KW_BLOCKAGE else self._specialized_regions.append(r)

        # Obtained the fixed rectangles from the netlist
        self._fixed = [] if netlist is None else netlist.fixed_rectangles()

        self._x, self._y = gather_boundaries(self.specialized_regions + self.blockages +
                                             self.fixed_regions + [self.bounding_box], self._epsilon)
        self._calculate_cell_matrix()
        self._calculate_ground_rectangles()
        self._check_rectangles()

    @property
    def netlist(self) -> Netlist | None:
        """Returns the netlist associated to the die"""
        return self._netlist

    @property
    def bounding_box(self) -> Rectangle:
        """Returns the bounding box of the die"""
        return self._die

    @property
    def width(self) -> float:
        """Returns the width of the die."""
        return self._die.shape.w

    @property
    def height(self) -> float:
        """Returns the height of the die."""
        return self._die.shape.h

    @property
    def ground_regions(self) -> list[Rectangle]:
        """Returns the list of ground regions not covered by fixed rectangles"""
        return self._ground_regions

    @property
    def specialized_regions(self) -> list[Rectangle]:
        """Returns the list of non-ground regions."""
        return self._specialized_regions

    @property
    def blockages(self) -> list[Rectangle]:
        """Returns the list of blockages."""
        return self._blockages

    @property
    def fixed_regions(self) -> list[Rectangle]:
        """Returns the list of fixed rectangles."""
        return self._fixed

    def split_refinable_regions(self, aspect_ratio: float, n: int = 1) -> None:
        """
        Splits the refinable rectangles such that all of them have an aspect_ratio smaller than or equal to
        a certain value. After that, rectangles are split until n rectangles are obtained. The rectangles correspond
        to the ground and non-ground rectangles. Fixed rectangles are neither modified nor counted. The aspect ratio
        is always >= 1, i.e., max(width/height, height/width)
        :param aspect_ratio: the maximum aspect ratio of the rectangles (must be greater than > sqrt(2))
        :param n: number of refinable rectangles that are required
        """
        assert n > 0
        assert aspect_ratio > 1.415, "Aspect ratio cannot be smaller than sqrt(2) to guarantee convergence"
        rects = split_rectangles(self.specialized_regions + self.ground_regions, aspect_ratio, n)
        self._specialized_regions, self._ground_regions = [], []
        for r in rects:
            if r.region == KW_GROUND:
                self._ground_regions.append(r)
            else:
                self._specialized_regions.append(r)

    def initial_grid(self, nrows: int, ncols: int) -> None:
        """
        Creates a matrix of refinable rectangles from a clean die (no specialized regions, no fixed modules,
        no blockages)
        :param nrows: number of rows of the grid
        :param ncols: number of columns of the grid
        """
        assert nrows > 0 and ncols > 0 and nrows + ncols > 1
        assert len(self.fixed_regions) == 0 and len(self.specialized_regions) == 0 and len(self.blockages) == 0,\
            "Cannot create a gridded die: it has blockages, fixed regions or specialized regions."
        assert len(self.ground_regions) == 1, "Cannot create a gridded die: it has more than one ground region."
        self._ground_regions = self._die.rectangle_grid(nrows, ncols)

    def floorplanning_rectangles(self) -> tuple[list[Rectangle], list[Rectangle]]:
        """
        Returns the two lists of rectangles usable for module allocation. The first list contains
        the rectangles that a refinable during allocation. The second list contains the rectangles
        that correspond to fixed modules.
        """
        return self.specialized_regions + self.ground_regions, self.fixed_regions

    def _cell_center(self, i: int, j: int) -> Point:
        """
        Returns the center of a cell
        :param i: row of the cell
        :param j: column of the cell
        :return: the center of the cell
        """
        return Point((self._x[i] + self._x[i + 1]) / 2, (self._y[j] + self._y[j + 1]) / 2)

    def _calculate_cell_matrix(self):
        """
        Calculates the matrix of cells. It indicates which cells are occupied by regions, blockages or fixed rectangles
        """
        self._cells = [[False] * (len(self._x) - 1) for _ in range(len(self._y) - 1)]
        for i in range(len(self._x) - 1):
            for j in range(len(self._y) - 1):
                p = self._cell_center(i, j)
                for r in self.specialized_regions + self.blockages + self.fixed_regions:
                    if r.point_inside(p):
                        self._cells[j][i] = True

    def _cell_inside_rectangle(self, i: int, j: int, r: Rectangle) -> bool:
        """
        Determines whether cell[i,j] is inside a rectangle
        :param i: row of the cell
        :param j: column of the cell
        :param r: rectantle
        :return: True if it is inside the rectangle, and False otherwise
        """
        return r.point_inside(self._cell_center(i, j))

    def _calculate_ground_rectangles(self):
        self._ground_regions = []
        all_rectangles: Set[GroundRegion] = self._find_all_ground_rectangles()
        while len(all_rectangles) > 0:
            self._ground_regions.append(self._find_best_rectangle(all_rectangles))

    def _find_all_ground_rectangles(self) -> Set[GroundRegion]:
        """
        Calculates all possible ground rectangles
        :return: the set of ground rectangles
        """
        all_regions: Set[GroundRegion] = set()  # Set of all rectangular regions
        for r in range(len(self._cells)):
            height = self._y[r + 1] - self._y[r]
            for c in range(len(self._cells[r])):
                if not self._cells[r][c]:
                    width = self._x[c + 1] - self._x[c]
                    area = width * height
                    ratio = height / width
                    if ratio < 1.0:
                        ratio = 1 / ratio
                    reg: GroundRegion = GroundRegion(r, r, c, c, area, ratio)
                    more_regions = self._expand_rectangle(reg)
                    all_regions |= more_regions
        return all_regions

    def _find_best_rectangle(self, ground_rectangles: Set[GroundRegion]) -> Rectangle | None:
        """
        Calculates the largest non-occupied rectangular region of the die
        :param ground_rectangles: set of possible ground rectangles
        :return: the largest region
        """

        assert len(ground_rectangles) > 0

        # Here we select the best rectangle. The criterion may be changed in the future
        max_value = -1.0
        best_reg: GroundRegion | None = None
        for reg in ground_rectangles:
            value = reg.area
            if reg.area > max_value:
                max_value = value
                best_reg = reg

        # Occupy the cells
        assert best_reg is not None
        for row in range(best_reg.rmin, best_reg.rmax + 1):
            for col in range(best_reg.cmin, best_reg.cmax + 1):
                self._cells[row][col] = True

        # Remove the rectangles touching the occupied cells
        for reg in list(ground_rectangles):
            if any(self._cells[row][col] for row in range(reg.rmin, reg.rmax + 1)
                   for col in range(reg.cmin, reg.cmax + 1)):
                ground_rectangles.remove(reg)

        x_center = (self._x[best_reg.cmin] + self._x[best_reg.cmax + 1]) / 2
        y_center = (self._y[best_reg.rmin] + self._y[best_reg.rmax + 1]) / 2
        width = self._x[best_reg.cmax + 1] - self._x[best_reg.cmin]
        height = self._y[best_reg.rmax + 1] - self._y[best_reg.rmin]
        kwargs = {KW_CENTER: Point(x_center, y_center), KW_SHAPE: Shape(width, height), KW_REGION: KW_GROUND}
        return Rectangle(**kwargs)

    def _expand_rectangle(self, r: GroundRegion) -> Set[GroundRegion]:
        """
        Expands a rectangle of regions and generates all the valid regions.
        The expansion is done by increasing rows and columns
        :param r: a ground region
        :return: the set of rectangles of ground regions
        """
        g_regions: Set[GroundRegion] = {r}
        pending: Deque[GroundRegion] = deque()
        pending.append(r)
        while len(pending) > 0:
            r = pending.popleft()
            if r.rmax < len(self._cells) - 1:  # Add one row
                row = r.rmax + 1
                valid = not any(self._cells[row][j] for j in range(r.cmin, r.cmax + 1))
                if valid:
                    height = self._y[r.rmax + 2] - self._y[r.rmin]
                    width = self._x[r.cmax + 1] - self._x[r.cmin]
                    area = height * width
                    ratio = height / width
                    if ratio < 1.0:
                        ratio = 1 / ratio
                    new_r = GroundRegion(r.rmin, r.rmax + 1, r.cmin, r.cmax, area, ratio)
                    if new_r not in g_regions:
                        g_regions.add(new_r)
                        pending.append(new_r)

            if r.cmax < len(self._cells[0]) - 1:  # Add one column
                col = r.cmax + 1
                valid = not any(self._cells[i][col] for i in range(r.rmin, r.rmax + 1))
                if valid:
                    height = self._y[r.rmax + 1] - self._y[r.rmin]
                    width = self._x[r.cmax + 2] - self._x[r.cmin]
                    area = height * width
                    ratio = height / width
                    if ratio < 1.0:
                        ratio = 1 / ratio
                    new_r = GroundRegion(r.rmin, r.rmax, r.cmin, r.cmax + 1, area, ratio)
                    if new_r not in g_regions:
                        g_regions.add(new_r)
                        pending.append(new_r)

        return g_regions

    def _check_rectangles(self) -> None:
        """
        Checks that the list of rectangles is correct, i.e., they do not overlap and the sum of the
        areas is equal to the area of the die. An assertion is raised of something is wrong.
        """
        all_rectangles = self.specialized_regions + self.ground_regions + self.blockages + self.fixed_regions
        die = Rectangle(center=Point(self.width / 2, self.height / 2), shape=Shape(self.width, self.height))

        # Check that all rectangles are inside
        for r in all_rectangles:
            bb = r.bounding_box
            assert die.point_inside(bb.ll) and die.point_inside(bb.ur), "Some rectangle is outside the die"

        # Check that no rectangles overlap
        pairs = list(combinations(all_rectangles, 2))
        for r1, r2 in pairs:
            assert not r1.overlap(r2), "Some rectangles overlap"

        # Check that the total area of the rectangles is equal to the area of the die
        area_rect = sum(r.area for r in all_rectangles)
        assert abs(area_rect - die.area) < self._epsilon, "Incorrect total area of rectangles"
