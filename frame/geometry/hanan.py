# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from collections.abc import Iterable
from dataclasses import dataclass
from geometry import Point, Rectangle


@dataclass
class HananCoords:
    """Class to represent the coordinates of the horizontal
    and vertical axes of a Hanan grid. Typically the lower-left corner
    has coordinates (0, 0), but it is not mandatory.
    The Hanan grid is a n x m matrix, where n=len(x)-1 and m=len(y)-1."""

    x: list[float]  # sorted list of x coordinates
    y: list[float]  # sorted list of y coordinates


class HananGrid:
    """Class to manipulate Hanan grids for floorplanning. The class is
    initialized with a set of rectangles that determine the boundaries of
    the grid cells."""

    _coords: HananCoords  # coordinates of the Hanan grid
    _centers: HananCoords  # centers of the cells of the Hanan grid

    def __init__(self, rect: Iterable[Rectangle] = list(), add_origin: bool = False):
        """Generates the coordinates of the Hanan grid from a set of rectangles.
        If add_origin, the coordinates (0,0) are also added."""

        bboxes = [r.bounding_box for r in rect]
        xcoords = set[float]()  # set of unique x coordinates
        ycoords = set[float]()  # set of unique y coordinates
        for r in rect:
            bb = r.bounding_box
            xcoords.update([float(bb.ll.x), float(bb.ur.x)])
            ycoords.update([float(bb.ll.y), float(bb.ur.y)])

        if add_origin:
            xcoords.add(0.0)
            ycoords.add(0.0)
        self._coords.x = list(xcoords)
        self._coords.y = list(ycoords)
        self._coords.x.sort()
        self._coords.y.sort()
        self._centers.x = [
            (self._coords.x[i] + self._coords.x[i + 1]) / 2
            for i in range(len(self._coords.x) - 1)
        ]
        self._centers.y = [
            (self._coords.y[i] + self._coords.y[i + 1]) / 2
            for i in range(len(self._coords.y) - 1)
        ]

    def hanan_cell_inside(self, i: int, j: int, r: Rectangle) -> bool:
        """Indicates whether a Hanan cell is inside a rectangle"""
        assert 0 <= i < len(self._centers.x) and 0 <= j < len(self._centers.y), (
            "Cell indices outside the boundaries of the Hanan grid"
        )
        return r.point_inside(Point(self._centers.x[i], self._centers.y[j]))
