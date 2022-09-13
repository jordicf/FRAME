# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Modules of a netlist
"""

import math
from frame.geometry.geometry import Point, Shape, Rectangle, create_stog
from frame.utils.keywords import KW_CENTER, KW_SHAPE, KW_MIN_SHAPE, KW_AREA, KW_FIXED, KW_HARD, KW_FLIP,\
    KW_GROUND, KW_RECTANGLES
from frame.utils.utils import valid_identifier, is_number


class Module:
    """
    Class to represent a module of the system
    """
    _name: str  # Name of the module
    _center: Point | None  # Center of the module (if defined)
    _min_shape: Shape | None  # min width and height
    _hard: bool  # Must be a hard module (but movable if not fixed)
    _fixed: bool  # Must be fixed in the layout
    _flip: bool  # May be flipped (only for hard modules, not fixed)
    # A module can be assigned to different layout areas (LUT, DSP, BRAM).
    # The following attribute indicate the area assigned to each region.
    # If no region specified, the default is assigned (Ground).
    _area_regions: dict[str, float]  # Area for each type of region
    # This is an attribute to store the whole area of the module. If not calculated, it
    # has a negative value
    _total_area: float
    # A module can be implemented with different rectangles. Here is the list of rectangles
    _rectangles: list[Rectangle]  # Rectangles of the module (if defined)
    _area_rectangles: float  # Total area of the rectangles (negative if not calculated).

    # Allocation of regions. This dictionary receives the name of a rectangle (from the die)
    # as key and stores the ratio of occupation of the rectangle by the module (a value in [0,1]).

    def __init__(self, name: str, **kwargs):
        """
        Constructor
        :param kwargs: name (str), center (Point), min_shape (Shape), area (float or dict), fixed (boolean)
        """
        self._name = name
        self._center = None
        self._min_shape = None
        self._hard = False
        self._fixed = False
        self._flip = False
        self._area_regions = {}
        self._total_area = -1
        self._rectangles = []
        self._area_rectangles = -1

        assert valid_identifier(name), "Incorrect module name"

        # Reading parameters and type checking
        for key, value in kwargs.items():
            assert key in [KW_CENTER, KW_MIN_SHAPE, KW_AREA, KW_HARD, KW_FIXED, KW_FLIP],\
                "Unknown module attribute"
            if key == KW_CENTER:
                assert isinstance(value, Point), "Incorrect point associated to the center of the module"
                self._center = value
            elif key == KW_MIN_SHAPE:
                assert isinstance(value, Shape), "Incorrect shape associated to the module"
                assert value.w >= 0, "Incorrect module width"
                assert value.h >= 0, "incorrect module height"
                self._min_shape = value
            elif key == KW_AREA:
                self._area_regions = self._read_region_area(value)
            elif key == KW_FIXED:
                assert isinstance(value, bool), "Incorrect value for fixed (should be a boolean)"
                self._fixed = value
            elif key == KW_HARD:
                assert isinstance(value, bool), "Incorrect value for hard (should be a boolean)"
                self._hard = value
            elif key == KW_FLIP:
                assert isinstance(value, bool), "Incorrect value for flip (should be a boolean)"
                self._flip = value
            else:
                assert False  # Should never happen

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Module):
            return NotImplemented
        return self._name == other._name

    @property
    def name(self) -> str:
        return self._name

    # Getter and setter for center
    @property
    def center(self) -> Point | None:
        return self._center

    @center.setter
    def center(self, p: Point) -> None:
        self._center = p

    # Getter and setter for min_shape
    @property
    def min_shape(self) -> Shape | None:
        return self._min_shape

    @min_shape.setter
    def min_shape(self, shape: Shape) -> None:
        self._min_shape = shape

    @property
    def is_hard(self) -> bool:
        return self._hard

    @property
    def is_fixed(self) -> bool:
        return self._fixed

    @is_fixed.setter
    def is_fixed(self, value: bool) -> None:
        if self._fixed != value:
            self._fixed = value
            for rectangle in self._rectangles:
                rectangle.fixed = value

    @property
    def is_soft(self) -> bool:
        return not self._hard

    @property
    def flip(self) -> bool:
        return self._flip

    # Getters for area
    def area(self, region: str | None = None) -> float:
        """Returns the area of a module associated to a region.
        If no region is specified, the total area is returned."""
        if region is None:
            if self._total_area < 0:
                self._total_area = sum(self._area_regions.values())
            return self._total_area
        if region not in self._area_regions:
            return 0
        return self._area_regions[region]

    @property
    def area_regions(self) -> dict[str, float]:
        """Returns the dictionary of regions and associated areas."""
        return self._area_regions

    @property
    def num_rectangles(self) -> int:
        """Returns the number of rectangles of the module."""
        return len(self.rectangles)

    @property
    def rectangles(self) -> list[Rectangle]:
        """Returns the list of rectangles of the module."""
        return self._rectangles

    @property
    def area_rectangles(self) -> float:
        """Returns the total area of the rectangles of the module"""
        if self._area_rectangles < 0:
            self._area_rectangles = sum(r.area for r in self.rectangles)
        return self._area_rectangles

    def clear_rectangles(self) -> None:
        """Removes all rectangles of a module"""
        self._rectangles = []

    def add_rectangle(self, r: Rectangle) -> None:
        """
        Adds a rectangle to the module
        :param r: the rectangle
        """
        self.rectangles.append(r)
        self._area_rectangles = -1

    def recenter_rectangles(self) -> None:
        """Recenters the rectangles of a hard module according to its center"""
        # Calculate current center
        assert self.is_hard and not self.is_fixed and self.center is not None
        area = sum(r.area for r in self.rectangles)
        x = sum(r.center.x*r.area for r in self.rectangles)/area
        y = sum(r.center.y*r.area for r in self.rectangles)/area
        inc_x, inc_y = self.center.x - x, self.center.y - y
        for r in self.rectangles:
            r.center.x += inc_x
            r.center.y += inc_y

    @staticmethod
    def _read_region_area(area: float | dict[str, float]) -> dict[str, float]:
        """Treats the area of a module as a float or as a dictionary of {region: float},
        where region is a string
        :param area: area of the region. If only one number is specified, the area is
        assigned to the ground region.
        :return: a dictionary with the area associated to each region.
        """
        if isinstance(area, (int, float)):
            float_area = float(area)
            assert float_area > 0, "Area must be positive"
            return {KW_GROUND: float_area}

        dict_area: dict[str, float] = {}
        assert isinstance(area, dict), "Invalid area specification"
        for region, a in area.items():
            assert valid_identifier(region), "Invalid region identifier"
            assert is_number(a) and a > 0, "Invalid value for area"
            dict_area[region] = float(a)
        return dict_area

    def setup(self) -> None:
        """
        Checking the consistency of the module. No area must have been defined for hard/fixed modules.
        For soft blocks, the area must have been defined. The rectangles for hard modules must not
        overlap. The first rectangle of hard blocks must be the trunk
        """

        assert not (self.is_fixed and not self.is_hard),\
            f"Inconsistent fixed module {self.name}. It should be also hard."

        assert not(self.flip and self.is_fixed), f"Fixed module {self.name} cannot be flipped."
        assert not(self.flip and not self.is_hard), f"Soft module {self.name} cannot be flipped."

        area_defined = len(self.area_regions) > 0
        assert self.is_hard or area_defined, f"No area defined for a soft module {self.name}."

        if self.is_hard:
            # Check that neither area, nor center nor min_shape are defined.
            # It also checks that at least has one rectangle
            assert not area_defined, f"Inconsistent hard module {self.name}: cannot specify area."
            assert self.center is None, f"Inconsistent hard module {self.name}: cannot specify center."
            assert self.min_shape is None,\
                f"Inconsistent hard module {self.name}: cannot specify min_shape."
            assert self.num_rectangles > 0, \
                f"Inconsistent hard module {self.name}: must have at least one rectangle."

            # Calculate the area of hard modules
            area = sum(r.area for r in self.rectangles)
            self._area_regions = {KW_GROUND: area}
            self._total_area = area

    @property
    def has_stog(self) -> bool:
        """
        Determines whether a module is a STOG
        :return: True if it has an associate STOG, and False otherwise
        """
        return self.num_rectangles > 0 and self.rectangles[0].location == Rectangle.StogLocation.TRUNK

    def create_square(self) -> None:
        """
        Creates a square for the module with the total area (and removes the previous rectangles)
        """
        assert self.center is not None, f"Cannot calculate square for module {self.name}. Missing center."
        area = self.area()
        assert area > 0, f"Cannot calculate square for module {self.name}. Area is zero."
        side = math.sqrt(area)
        self._rectangles = []
        self.add_rectangle(Rectangle(**{KW_CENTER: self.center, KW_SHAPE: Shape(side, side)}))

    def create_stog(self) -> bool:
        """
        Defines the locations of the rectangles in a Single-Trunk Orthogon (STOG). It returns true if the module
        can be represented as a STOG of the rectangles. It also defines the locations of
        the rectangles in the STOG. If no polygon is found, all rectangles have the NO_POLYGON location
        :return: True if a STOG has been identified, and False otherwise
        """
        return create_stog(self.rectangles)

    def calculate_center_from_rectangles(self) -> Point:
        """
        Calculates the center from the rectangles. It raises an exception in case there are no rectangles.
        :return: the center of the module .
        """
        assert self.num_rectangles > 0, f"No rectangles in module {self.name}"
        sum_x, sum_y, area = 0.0, 0.0, 0.0
        for r in self.rectangles:
            sum_x += r.area * r.center.x
            sum_y += r.area * r.center.y
            area += r.area
        self.center = Point(sum_x / area, sum_y / area)
        return self.center

    def __str__(self) -> str:
        s = f"{self.name}: {KW_AREA}={self.area_regions} {KW_CENTER}={self.center}"
        s += f" {KW_MIN_SHAPE}={self.min_shape}"
        if self.num_rectangles == 0:
            return s
        s += f" {KW_RECTANGLES}=["
        for r in self.rectangles:
            s += f"({str(r)})"
        s += "]"
        return s
