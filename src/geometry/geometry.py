"""
Module to represent points, shapes and rectangles
"""

from typing import NamedTuple, Tuple, Any

from keywords import KW_FIXED, KW_CENTER, KW_SHAPE, KW_REGION, KW_NAME, KW_GROUND
from utils import valid_identifier


# Representation of a point
class Point(NamedTuple):
    x: float
    y: float


# Representation of a shape (width and height)
class Shape(NamedTuple):
    w: float
    h: float


class Rectangle:
    """
    A class to represent a rectilinear rectangle
    """

    def __init__(self, **kwargs: Any):
        """
        Constructor
        :param kwargs: center (Point), shape (Shape), fixed (bool), region (str)
        """

        # Attributes
        self._center: Point = Point(-1, -1)     # Center of the rectangle
        self._shape: Shape = Shape(-1, -1)      # Shape: width and height
        self._fixed: bool = False               # Is the rectangle fixed?
        self._region: str = KW_GROUND           # Region of the layout to which the rectangle belongs to
        self._name: str = ""                    # Name of the rectangle

        # Reading parameters and type checking
        for key, value in kwargs.items():
            assert key in [KW_CENTER, KW_SHAPE, KW_FIXED, KW_REGION, KW_NAME], "Unknown rectangle attribute"
            if key == KW_CENTER:
                assert isinstance(value, Point), "Incorrect point associated to the center of the rectangle"
                self._center = value
            elif key == KW_SHAPE:
                assert isinstance(value, Shape), "Incorrect shape associated to the rectangle"
                assert value.w > 0, "Incorrect rectangle width"
                assert value.h > 0, "Incorrect rectangle height"
                self._shape = value
            elif key == KW_FIXED:
                assert isinstance(value, bool), "Incorrect value for fixed (should be a boolean)"
                self._fixed = value
            elif key == KW_REGION:
                assert valid_identifier(value), \
                    "Incorrect value for region (should be a valid string)"
                self._region = value
            elif key == KW_NAME:
                assert isinstance(value, str), "Incorrect value for rectangle"
                self._name = value
            else:
                assert False  # Should never happen

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
    def bounding_box(self) -> Tuple[Point, Point]:
        """
        :return: a tuple ((xmin, ymin), (xmax, ymax))
        """
        half_w, half_h = self.shape.w / 2, self.shape.h / 2
        xmin, xmax = self.center.x - half_w, self.center.x + half_w
        ymin, ymax = self.center.y - half_h, self.center.y + half_h
        return Point(xmin, ymin), Point(xmax, ymax)

    @property
    def area(self) -> float:
        return self._shape.w * self._shape.h

    @property
    def fixed(self) -> bool:
        return self._fixed

    @property
    def region(self) -> str:
        return self._region

    @region.setter
    def region(self, region: str) -> None:
        self._region = region

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name

    def inside(self, p: Point) -> bool:
        """
        Checks whether a point is inside the rectangle
        :param p: the point
        :return: True if inside, False otherwise
        """
        bb = self.bounding_box
        return bb[0].x <= p.x <= bb[1].x and bb[0].y <= p.y <= bb[1].y

    def overlap(self, r: 'Rectangle') -> bool:
        """
        Checks whether two rectangles overlap. They are considered not to overlap if they touch each other
        :param r: the other rectangle.
        :return: True if they overlap, and False otherwise.
        """
        ll1, ur1 = self.bounding_box
        ll2, ur2 = r.bounding_box
        if ur1.x <= ll2.x or ur2.x <= ll1.x:
            return False
        return ur1.y > ll2.y and ur2.y > ll1.y

    def __str__(self) -> str:
        """
        :return: string representation of the rectangle
        """
        s = f"{KW_CENTER}={self.center} {KW_SHAPE}={self.shape} {KW_REGION}={self.region}"
        if self.fixed:
            s += f" {KW_FIXED}"
        return s
