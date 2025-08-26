# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
from dataclasses import dataclass
from frame.geometry.geometry import Point, Rectangle


@dataclass
class HananCoords:
    """Class to represent the coordinates of the horizontal
    and vertical axes of a Hanan grid. Typically the lower-left corner
    has coordinates (0, 0), but it is not mandatory.
    The Hanan grid is a n x m matrix, where n=len(x)-1 and m=len(y)-1."""

    x: npt.NDArray[np.float64]  # sorted list of x coordinates
    y: npt.NDArray[np.float64]  # sorted list of y coordinates


@dataclass
class GridRectangle:
    """Class to represent a rectangle in a Hanan grid."""

    row_min: int  # lower row index of the rectangle
    row_max: int  # upper row index of the rectangle
    col_min: int  # lower column index of the rectangle
    col_max: int  # upper column index of the rectangle


class HananGrid:
    """Class to manipulate Hanan grids for floorplanning. The class is
    initialized with a set of rectangles that determine the boundaries of
    the grid cells."""

    _coords: HananCoords  # coordinates of the Hanan grid
    _centers: HananCoords  # centers of the cells of the Hanan grid
    _num_rects: npt.NDArray[np.int32]  # number of rectangles occupying each cell

    def __init__(self, rect: Iterable[Rectangle] = list(), add_origin: bool = False):
        """Generates the coordinates of the Hanan grid from a set of rectangles.
        If add_origin, the coordinates (0,0) are also added."""

        # we punt coordinates on sets to make values unique
        xcoords = set[float]()  # set of unique x coordinates
        ycoords = set[float]()  # set of unique y coordinates
        for r in rect:
            bb = r.bounding_box
            xcoords.update([float(bb.ll.x), float(bb.ur.x)])
            ycoords.update([float(bb.ll.y), float(bb.ur.y)])

        if add_origin:
            xcoords.add(0.0)
            ycoords.add(0.0)
        self._coords.x = np.sort(list(xcoords))
        self._coords.y = np.sort(list(ycoords))

        self._centers.x = np.array(
            [
                (self._coords.x[i] + self._coords.x[i + 1]) / 2
                for i in range(self._coords.x.size - 1)
            ]
        )
        self._centers.y = np.array(
            [
                (self._coords.y[i] + self._coords.y[i + 1]) / 2
                for i in range(self._coords.y.size - 1)
            ]
        )

        self._num_rects = np.zeros(
            (self._centers.x.size, self._centers.y.size), dtype=np.int32
        )

        # This is not very efficient, but it is simple
        for r in rect:
            for i in range(self._centers.x.size):
                for j in range(self._centers.y.size):
                    if self.hanan_cell_inside(i, j, r):
                        self._num_rects[i, j] += 1

    def hanan_cell_inside(self, i: int, j: int, r: Rectangle) -> bool:
        """Indicates whether a Hanan cell is inside a rectangle"""
        assert 0 <= i < self._centers.x.size and 0 <= j < self._centers.y.size, (
            "Cell indices outside the boundaries of the Hanan grid"
        )
        return r.point_inside(Point(self._centers.x[i], self._centers.y[j]))
