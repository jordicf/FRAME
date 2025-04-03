# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Module to represent points, shapes and rectangles
"""

from collections import deque
from enum import Enum
import heapq
import math
from typing import Any, Union, Sequence, Optional
from dataclasses import dataclass, field

from frame.utils.keywords import KW_FIXED, KW_HARD, KW_CENTER, KW_SHAPE, KW_REGION, KW_NAME, \
    KW_GROUND, KW_BLOCKAGE
from frame.utils.utils import valid_identifier, almost_eq

RectDescriptor = tuple[float, float, float, float, str]  # (x,y,w,h, region)


class Point:
    """
    A class to represent two-dimensional points and operate with them
    """

    _x: float  # x coordinate
    _y: float  # y coordinate

    def __init__(self, x: float = 0, y: float = 0) -> None:
        """
        Constructor of a Point.
        """
        self._x, self._y = x, y

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float):
        self._x = value

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: float):
        self._y = value

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        assert isinstance(other, Point)
        return self.x == other.x and self.y == other.y

    def __neg__(self) -> 'Point':
        """Return -self."""
        return Point(-self.x, -self.y)

    def __add__(self, other: 'Point') -> 'Point':
        """Return self + other."""
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        """Return self - other."""
        return Point(self.x - other.x, self.y - other.y)

    def __and__(self, other: 'Point') -> float:
        """Dot product between self and other."""
        return self.x * other.x + self.y * other.y

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)

    def __str__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    __repr__ = __str__

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class Shape:
    """
    A class to represent a two-dimensional rectilinear shape (width and height)
    """
    w: float
    h: float


@dataclass
class AspectRatio:
    """
    A class to represent the aspect ratio of a module or a rectangle (interval of width/height)
    """
    min_wh: float
    max_wh: float


@dataclass
class BoundingBox:
    """
    A class to represent a rectangle using a bounding box
    """
    ll: Point
    ur: Point


class Rectangle:
    """
    A class to represent a rectilinear rectangle
    """

    class StogLocation(Enum):
        """Class to represent the location of a rectangle in a Single-Trunk Orthogon (STOG).
        The values NSEW represent the location with regard to the trunk. If a rectangle is not
        in a STOG, the value NO_POLYGON is assigned"""
        TRUNK = 1
        NORTH = 2
        SOUTH = 3
        EAST = 4
        WEST = 5
        NO_POLYGON = 6

    _distance_epsilon: float = -1.0  # epsilon used for distances
    _area_epsilon: float = -1.0  # epsilon used for area

    def __init__(self, **kwargs: Any):
        """
        Constructor
        :param kwargs: center (Point), shape (Shape), fixed (bool), region (str)
        """

        # Attributes
        self._center: Point = Point(-1, -1)  # Center of the rectangle
        self._shape: Shape = Shape(-1, -1)  # Shape: width and height
        self._fixed: bool = False  # Is the rectangle fixed?
        self._hard: bool = False  # Is the rectangle hard?
        # Region of the layout to which the rectangle belongs to
        self._region: str = KW_GROUND
        self._location: Rectangle.StogLocation = Rectangle.StogLocation.NO_POLYGON

        # Reading parameters and type checking
        for key, value in kwargs.items():
            assert key in [KW_CENTER, KW_SHAPE, KW_FIXED, KW_HARD, KW_REGION, KW_NAME], \
                "Unknown rectangle attribute"
            if key == KW_CENTER:
                assert isinstance(
                    value, Point), "Incorrect point associated to the center of the rectangle"
                self._center = value
            elif key == KW_SHAPE:
                assert isinstance(
                    value, Shape), "Incorrect shape associated to the rectangle"
                assert value.w > 0, "Incorrect rectangle width"
                assert value.h > 0, "Incorrect rectangle height"
                self._shape = value
            elif key == KW_FIXED:
                assert isinstance(
                    value, bool), "Incorrect value for fixed (should be a boolean)"
                self._fixed = value
            elif key == KW_HARD:
                assert isinstance(
                    value, bool), "Incorrect value for hard (should be a boolean)"
                self._hard = value
            elif key == KW_REGION:
                assert valid_identifier(value) or value == KW_BLOCKAGE, \
                    "Incorrect value for region (should be a valid string)"
                self._region = value
            elif key == KW_NAME:
                assert isinstance(value, str), "Incorrect value for rectangle"
                self._name = value
            else:
                assert False  # Should never happen

    @staticmethod
    def epsilon_defined() -> bool:
        """Indicates whether epsilon has been defined for the class"""
        return Rectangle._distance_epsilon >= 0

    @staticmethod
    def set_epsilon(distance_epsilon: float, area_epsilon: float = -1) -> None:
        """
        Defines the epsilons for float comparisons in distance an area
        :param distance_epsilon: epsilon used for distances
        :param area_epsilon: epsilon used for area. If negative, sqrt(distance_epsilon) is taken
        """
        Rectangle._distance_epsilon = distance_epsilon
        Rectangle._area_epsilon = area_epsilon if area_epsilon >= 0 else math.sqrt(
            distance_epsilon)

    @staticmethod
    def undefine_epsilon() -> None:
        """Makes epsilon undefined for the class"""
        Rectangle._distance_epsilon = Rectangle._area_epsilon = -1.0

    @staticmethod
    def distance_epsilon() -> float:
        """Returns the epsilon used for distances"""
        assert Rectangle._distance_epsilon >= 0, "Undefined epsilon for Rectangle"
        return Rectangle._distance_epsilon

    @staticmethod
    def area_epsilon() -> float:
        """Returns the epsilon used for area"""
        assert Rectangle._area_epsilon >= 0, "Undefined epsilon for rectangle"
        return Rectangle._area_epsilon

    # Getter and setter for center
    @property
    def center(self) -> Point:
        return self._center

    @center.setter
    def center(self, p: Point) -> None:
        self._center = p

    # Getter and setter for shape
    @property
    def shape(self) -> Shape:
        return self._shape

    @shape.setter
    def shape(self, shape) -> None:
        self._shape = shape

    @property
    def fixed(self) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(self, value: bool) -> None:
        self._fixed = value

    @property
    def hard(self) -> bool:
        return self._hard

    @hard.setter
    def hard(self, value: bool) -> None:
        self._hard = value

    @property
    def region(self) -> str:
        return self._region

    @region.setter
    def region(self, region: str) -> None:
        self._region = region

    @property
    def aspect_ratio(self) -> float:
        assert self.shape.w > 0
        ar = self.shape.h / self.shape.w
        if ar < 1:
            ar = 1.0 / ar
        return ar

    @property
    def location(self) -> StogLocation:
        return self._location

    @location.setter
    def location(self, loc: StogLocation) -> None:
        self._location = loc

    def duplicate(self) -> 'Rectangle':
        """
        Creates a duplication of the rectangle
        :return: the rectangle
        """
        return Rectangle(**{KW_CENTER: self.center, KW_SHAPE: self.shape, KW_FIXED: self.fixed,
                            KW_HARD: self.hard, KW_REGION: self.region})

    @property
    def bounding_box(self) -> BoundingBox:
        """
        Returns the bounding box of the rectangle
        """
        half_w, half_h = self.shape.w / 2, self.shape.h / 2
        xmin, xmax = self.center.x - half_w, self.center.x + half_w
        ymin, ymax = self.center.y - half_h, self.center.y + half_h
        return BoundingBox(ll=Point(xmin, ymin), ur=Point(xmax, ymax))

    @property
    def vector_spec(self) -> RectDescriptor:
        """Returns a vector specification of the rectangle [x, y, w, h, region]"""
        return self.center.x, self.center.y, self.shape.w, self.shape.h, self.region

    @property
    def area(self) -> float:
        return self._shape.w * self._shape.h

    def point_inside(self, p: Point) -> bool:
        """
        Checks whether a point is inside the rectangle
        :param p: the point
        :return: True if inside, False otherwise
        """
        bb = self.bounding_box
        return bb.ll.x <= p.x <= bb.ur.x and bb.ll.y <= p.y <= bb.ur.y

    def is_inside(self, r: 'Rectangle') -> bool:
        """
        Checks whether the rectangle is inside another rectangle
        :param r: the other rectangle
        :return: True if inside, False otherwise
        """
        bb = self.bounding_box
        # It is a rectangle
        bbr = r.bounding_box
        return bb.ll.x >= bbr.ll.x and bb.ll.y >= bbr.ll.y and bb.ur.x <= bbr.ur.x and bb.ur.y <= bbr.ur.y

    def touches(self, r: 'Rectangle') -> bool:
        """Checks whether the two rectangles touch each other according to some distance tolerance
        :param r: the other rectangle
        :return: True if they touch each other, and False otherwise
        """
        bb_self, bb_r = self.bounding_box, r.bounding_box
        epsilon = Rectangle.distance_epsilon()
        return bb_self.ll.x <= bb_r.ur.x + epsilon and bb_r.ll.x <= bb_self.ur.x + epsilon \
            and bb_self.ll.y <= bb_r.ur.y + epsilon and bb_r.ll.y <= bb_self.ur.y + epsilon

    def overlap(self, r: 'Rectangle') -> bool:
        """
        Checks whether two rectangles overlap. They are considered not to overlap if they touch each other.
        If the overlapping area is smaller than epsilon, they are considered not to overlap
        :param r: the other rectangle
        :return: True if they overlap, and False otherwise
        """
        return self.area_overlap(r) > Rectangle.area_epsilon()

    def area_overlap(self, r: 'Rectangle') -> float:
        """
        Returns the area overlap between the two rectangles
        :param r: the other rectangle
        :return: the area overlap
        """
        bb1 = self.bounding_box
        bb2 = r.bounding_box
        minx = max(bb1.ll.x, bb2.ll.x)
        maxx = min(bb1.ur.x, bb2.ur.x)
        if minx >= maxx:
            return 0.0
        miny = max(bb1.ll.y, bb2.ll.y)
        maxy = min(bb1.ur.y, bb2.ur.y)
        if miny >= maxy:
            return 0.0
        return (maxx - minx) * (maxy - miny)

    def find_location(self, r: 'Rectangle') -> StogLocation:
        """Defines the location of a rectangle with regard to the trunk (self)
        :param r: the rectangle that must be located
        :return: the location
        """

        # If they overlap, it cannot be a branch
        if self.area_overlap(r) > Rectangle.area_epsilon():
            return Rectangle.StogLocation.NO_POLYGON

        epsilon = Rectangle.distance_epsilon()

        bb_self = self.bounding_box
        bb_r = r.bounding_box

        # Let us first find the common side and then check the interval
        if almost_eq(bb_self.ur.y, bb_r.ll.y, epsilon):
            loc = Rectangle.StogLocation.NORTH
        elif almost_eq(bb_self.ll.y, bb_r.ur.y, epsilon):
            loc = Rectangle.StogLocation.SOUTH
        elif almost_eq(bb_self.ur.x, bb_r.ll.x, epsilon):
            loc = Rectangle.StogLocation.EAST
        elif almost_eq(bb_self.ll.x, bb_r.ur.x, epsilon):
            loc = Rectangle.StogLocation.WEST
        else:
            return Rectangle.StogLocation.NO_POLYGON

        # Check the intervals
        if loc in [Rectangle.StogLocation.NORTH, Rectangle.StogLocation.SOUTH]:
            return loc if bb_r.ll.x > bb_self.ll.x - epsilon and bb_r.ur.x < bb_self.ur.x + epsilon \
                else Rectangle.StogLocation.NO_POLYGON

        # West or East
        return loc if bb_r.ll.y > bb_self.ll.y - epsilon and bb_r.ur.y < bb_self.ur.y + epsilon \
            else Rectangle.StogLocation.NO_POLYGON

    def split_horizontal(self, x: float = -1) -> tuple['Rectangle', 'Rectangle']:
        """
        Splits the rectangle horizontally cutting by x. If x is negative, the rectangle is split into two halves
        :param x: the x-cut
        :return: two rectangles
        """
        if x < 0:
            x = self.center.x
        bb = self.bounding_box
        assert bb.ll.x < x < bb.ur.x
        c1 = Point((bb.ll.x + x) / 2, self.center.y)
        sh1 = Shape(x - bb.ll.x, self.shape.h)
        c2 = Point((bb.ur.x + x) / 2, self.center.y)
        sh2 = Shape(self.shape.w - sh1.w, self.shape.h)
        r1, r2 = self.duplicate(), self.duplicate()
        r1.center, r1.shape = c1, sh1
        r2.center, r2.shape = c2, sh2
        return r1, r2

    def split_vertical(self, y: float = -1) -> tuple['Rectangle', 'Rectangle']:
        """
        Splits the rectangle vertically cutting by y. If y is negative, the rectangle is split into two halves
        :param y: the y-cut
        :return: two rectangles
        """
        if y < 0:
            y = self.center.y
        bb = self.bounding_box
        assert bb.ll.y < y < bb.ur.y
        c1 = Point(self.center.x, (bb.ll.y + y) / 2)
        sh1 = Shape(self.shape.w, y - bb.ll.y)
        c2 = Point(self.center.x, (bb.ur.y + y) / 2)
        sh2 = Shape(self.shape.w, self.shape.h - sh1.h)
        r1, r2 = self.duplicate(), self.duplicate()
        r1.center, r1.shape = c1, sh1
        r2.center, r2.shape = c2, sh2
        return r1, r2

    def split(self) -> tuple['Rectangle', 'Rectangle']:
        """
        Splits the rectangle into two rectangles. The splitting reduces the largest dimension
        :return: The two rectangles
        """
        return self.split_vertical() if self.shape.h > self.shape.w else self.split_horizontal()

    def x_cuttable(self, x: float, ratio: float = 0.01) -> bool:
        """
        Checks whether the rectangle can be cut vertically at coordinate x in a way that
        the smallest chunk is larger than ratio*area (e.g. 0.01 means 1%)
        :param x: coordinate of the horizontal cut
        :param ratio: ratio of the rectangle that defines the minimum area of the
        smallest rectangle after the cut
        :return: True if cuttable, False otherwise
        """
        bb = self.bounding_box
        if x <= bb.ll.x or x >= bb.ur.x:
            return False
        return min(x - bb.ll.x, bb.ur.x - x) > ratio * self.shape.h

    def y_cuttable(self, y: float, ratio: float = 0.01) -> bool:
        """
        Checks whether the rectangle can be cut horizontally at coordinate y in a way that
        the smallest chunk is larger than ratio*area (e.g. 0.01 means 1%)
        :param y: coordinate of the vertical cut
        :param ratio: ratio of the rectangle that defines the minimum area of the
        smallest rectangle after the cut
        :return: True if cuttable, False otherwise
        """
        bb = self.bounding_box
        if y <= bb.ll.y or y >= bb.ur.y:
            return False
        return min(y - bb.ll.y, bb.ur.y - y) > ratio * self.shape.w

    def rectangle_grid(self, nrows: int, ncols: int) -> list['Rectangle']:
        """
        Generates a grid of nrows x ncols rectangles of the same size starting from the original
        rectangle.
        :param nrows: number of rows of the grid
        :param ncols: number of columns of the grid
        :return: the list of rectangles
        """
        assert nrows > 0 and ncols > 0
        x_step = self.shape.w / ncols
        y_step = self.shape.h / nrows
        x_init = self.center.x - self.shape.w / 2 + x_step / 2
        y_init = self.center.y - self.shape.h / 2 + y_step / 2
        grid: list[Rectangle] = []
        shape = Shape(x_step, y_step)
        for row in range(nrows):
            for col in range(ncols):
                r = self.duplicate()
                r.center = Point(x_init + col * x_step, y_init + row * y_step)
                r.shape = shape
                grid.append(r)
        return grid

    def __mul__(self, other: 'Rectangle') -> Optional['Rectangle']:
        """
        Calculates the intersection of two rectangles and returns another rectangle (or None if no intersection).
        If the rectangles belong to different regions, None is returned
        :param other: The other rectangle
        :return: a rectangle representing the intersection (or None if no intersection)
        """
        if self.region != other.region:
            return None
        bb1 = self.bounding_box
        bb2 = other.bounding_box
        minx = max(bb1.ll.x, bb2.ll.x)
        maxx = min(bb1.ur.x, bb2.ur.x)
        width = maxx - minx
        if width <= 0:
            return None
        miny = max(bb1.ll.y, bb2.ll.y)
        maxy = min(bb1.ur.y, bb2.ur.y)
        height = maxy - miny
        if height <= 0:
            return None
        center = Point(minx + width / 2, miny + height / 2)
        r = self.duplicate()
        r.center, r.shape = center, Shape(width, height)
        return r

    def __eq__(self, other: Any) -> bool:
        """
        Checks whether two rectangles are the same (same center, same shape)
        :param other: the other rectangle (potentially another type of object)
        :return: True if equal, and False otherwise
        """
        if not isinstance(other, Rectangle):
            return False
        return self.center == other.center and self.shape == other.shape and self.region == other.region

    def __str__(self) -> str:
        """
        :return: string representation of the rectangle
        """
        s = f"({KW_CENTER}={self.center}, {KW_SHAPE}={self.shape}"
        if self.region != KW_GROUND:
            s += f", {KW_REGION}={self.region}"
        if self.fixed:
            s += f", {KW_FIXED}"
        s += ")"
        return s

    __repr__ = __str__


def parse_yaml_rectangle(r: Sequence[float | int | str],
                         fixed: bool = False, hard: bool = False) -> Rectangle:
    """Parses a rectangle
    :param r: a YAML description of the rectangle (a tuple or list with 4 numeric values (x, y, w, h)).
    Optionally, it may contain a fifth parameter (string) specifying a region
    :param fixed: Indicates whether the rectangle should be fixed
    :param hard: Indicates whether the rectangle should be hard
    :return: a rectangle
    """

    if isinstance(r, list):
        r = tuple(r)
    assert isinstance(r, tuple) and 4 <= len(
        r) <= 5, "Incorrect format for rectangle"
    for i in range(4):
        x = r[i]
        assert isinstance(
            x, (int, float)) and x >= 0, "Incorrect value for rectangle"
    if len(r) == 5:
        assert isinstance(r[4], str) and valid_identifier(r[4])

    # Hard or fixed rectangles must not be assigned to any region
    assert len(r) == 4 or not (
        fixed or hard), "Hard rectangles cannot be assigned to any region"

    assert isinstance(r[0], (int, float)) and isinstance(r[1], (int, float)) and \
        isinstance(r[2], (int, float)) and isinstance(r[3], (int, float))
    kwargs = {KW_CENTER: Point(r[0], r[1]), KW_SHAPE: Shape(r[2], r[3]),
              KW_FIXED: fixed, KW_HARD: hard}
    if len(r) == 5:
        kwargs[KW_REGION] = r[4]
    return Rectangle(**kwargs)


def gather_boundaries(rectangles: list[Rectangle]) -> tuple[list[float], list[float]]:
    """
    Gathers the x and y coordinates of the sides of a list of rectangles
    :param rectangles: list of rectangles
    :return: the list of x and y coordinates, sorted in ascending order
    """
    epsilon = Rectangle.distance_epsilon()
    x, y = [], []
    for r in rectangles:
        bb = r.bounding_box
        x.append(bb.ll.x)
        x.append(bb.ur.x)
        y.append(bb.ll.y)
        y.append(bb.ur.y)
    x.sort()
    y.sort()
    # Remove duplicates
    uniq_x: list[float] = []
    for i, val in enumerate(x):
        if i == 0 or val > uniq_x[-1] + epsilon:
            uniq_x.append(float(val))
    uniq_y: list[float] = []
    for i, val in enumerate(y):
        if i == 0 or val > uniq_y[-1] + epsilon:
            uniq_y.append(float(val))
    return uniq_x, uniq_y


def split_rectangles(rectangles: list[Rectangle], aspect_ratio: float, n: int) -> list[Rectangle]:
    """
    Splits the rectangles until n rectangles are obtained. The splitting is done on the
    largest rectangles of the list
    :param rectangles: list of rectangles
    :param aspect_ratio: maximum aspect ratio (must be greater than sqrt(2))
    :param n: number of required rectangles
    :return: the final rectangles
    """

    @dataclass(order=True)
    class PrioritizedRectangle:
        """To represent rectangles ordered by area"""
        area: float  # area of the rectangle (negative area to sort by largest)
        rect: Rectangle = field(compare=False)

    assert n > 0
    assert aspect_ratio > 1.415, "Aspect ratio cannot be smaller than sqrt(2) to guarantee convergence"

    # First split rectangles with large aspect ratio
    q: deque[Rectangle] = deque(rectangles)
    heap: list[PrioritizedRectangle] = []
    while len(q) > 0:
        r = q.pop()
        if r.aspect_ratio > aspect_ratio:
            q.extend(r.split())
        else:
            heap.append(PrioritizedRectangle(-r.area, r))

    # Do we have sufficient rectangles?
    if len(heap) >= n:
        return [prio_rect.rect for prio_rect in heap]

    # If not, let us split the largest rectangles (heap prioritized by area, the largest first)
    heapq.heapify(heap)
    while len(heap) < n:
        area_rect: PrioritizedRectangle = heapq.heappop(heap)
        r1, r2 = area_rect.rect.split()
        heapq.heappush(heap, PrioritizedRectangle(-r1.area, r1))
        heapq.heappush(heap, PrioritizedRectangle(-r2.area, r2))
    return [prio_rect.rect for prio_rect in heap]


def create_stog(rectangles: list[Rectangle]) -> bool:
    """
    Identifies the rectangles of a Single-Trunk Orthogon. At the end of the function, the location of
    each rectangle is defined (in case the STOG has been identified). In case more than one rectangle
    can be a trunk, the one with the largest area is selected. The selected trunk is put at the front of the list.
    If no STOG can be identified, it returns False
    :param rectangles: list of rectangles of the polygon
    :return: True if the STOG is identified, and False otherwise
    """
    assert len(rectangles) > 0

    if len(rectangles) == 1:
        rectangles[0].location = Rectangle.StogLocation.TRUNK
        return True

    for r in rectangles:
        r.location = Rectangle.StogLocation.NO_POLYGON

    best_trunk = -1
    for i, trunk in enumerate(rectangles):
        # Check if it is a good candidate
        if best_trunk >= 0 and trunk.area <= rectangles[best_trunk].area:
            break
        if all(r == trunk or trunk.find_location(r) != Rectangle.StogLocation.NO_POLYGON for r in rectangles):
            best_trunk = i

    if best_trunk < 0:
        return False

    # swap to put the trunk in front of the list
    rectangles[0], rectangles[best_trunk] = rectangles[best_trunk], rectangles[0]
    rectangles[0].location = Rectangle.StogLocation.TRUNK
    # Define the location for the rest of rectangles
    for i in range(1, len(rectangles)):
        rectangles[i].location = rectangles[0].find_location(rectangles[i])

    return True
