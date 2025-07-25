# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Modules of a netlist
"""

import math
from typing import Optional
from frame.geometry.geometry import Point, Shape, AspectRatio, Rectangle, create_strop
from frame.utils.keywords import KW
from frame.utils.utils import valid_identifier, is_number


class Module:
    """
    Class to represent a module of the system
    """

    _name: str  # Name of the module
    _center: Optional[Point]  # Center of the module (if defined)
    _aspect_ratio: AspectRatio | None  # interval of the aspect ratio
    _terminal: bool  # It is a terminal
    _hard: bool  # Must be a hard module (but movable if not fixed)
    _fixed: bool  # Must be fixed in the layout
    _flip: bool  # May be flipped (only for hard modules, not fixed)
    # A module can be assigned to different layout areas (LUT, DSP, BRAM).
    # The following attribute indicate the area assigned to each region.
    # If no region specified, the default is assigned (Ground).
    _area_regions: dict[str, float]  # Area for each type of region
    # This is an attribute to store the whole area of the module.
    # If not calculated, it has a negative value
    _total_area: float
    # A module can be implemented with different rectangles.
    # Here is the list of rectangles
    _rectangles: list[Rectangle]  # Rectangles of the module (if defined)
    # Total area of the rectangles (negative if not calculated).
    _area_rectangles: float

    # Allocation of regions. This dictionary receives the name of a rectangle
    # (from the die) as key and stores the ratio of occupation of the
    # rectangle by the module (a value in [0,1]).

    def __init__(self, name: str, **kwargs):
        """
        Constructor
        :param kwargs: name (str), center (Point), aspect_ratio (AspectRatio),
                       area (float or dict),
        hard (boolean), fixed (boolean), terminal (boolean)
        """
        self._name = name
        self._center = None
        self._aspect_ratio = None
        self._terminal = False
        self._hard = False
        self._fixed = False
        self._flip = False
        self._area_regions = {}
        self._total_area = -1
        self._rectangles = list[Rectangle]()
        self._area_rectangles = -1

        assert valid_identifier(name), "Incorrect module name"

        # Reading parameters and type checking
        for key, value in kwargs.items():
            match key:
                case KW.CENTER:
                    assert isinstance(value, Point), (
                        f"Module {name}: incorrect point associated to the center"
                    )
                    self._center = value
                case KW.ASPECT_RATIO:
                    assert isinstance(value, AspectRatio), (
                        f"Module {name}: incorrect aspect ratio"
                    )
                    assert 0 <= value.min_wh <= 1.0, (
                        f"Module {name}: incorrect aspect ratio"
                    )
                    assert value.max_wh >= 1.0, f"Module {name}: incorrect aspect ratio"
                    self._aspect_ratio = value
                case KW.AREA:
                    self._area_regions = self._read_region_area(value)
                case KW.FIXED:
                    assert isinstance(value, bool), (
                        f"Module {name}: incorrect value for fixed (should be a boolean)"
                    )
                    self._fixed = value
                    self._hard = value
                case KW.HARD:
                    assert KW.FIXED not in kwargs, (
                        f"Module {name}: {KW.FIXED} and {KW.HARD} are mutually exclusive"
                    )
                    assert isinstance(value, bool), (
                        f"Module {name}: incorrect value for hard (should be a boolean)"
                    )
                    self._hard = value
                case KW.FLIP:
                    assert isinstance(value, bool), (
                        f"Module {name}: incorrect value for flip (should be a boolean)"
                    )
                    self._flip = value
                case KW.TERMINAL:
                    assert KW.AREA not in kwargs, (
                        f"Module {name}: terminal cannot have area"
                    )
                    assert KW.ASPECT_RATIO not in kwargs, (
                        f"Module {name}: terminal cannot have aspect ratio"
                    )
                    assert KW.FLIP not in kwargs, (
                        f"Module {name}: terminal cannot have flip attribute"
                    )
                    assert isinstance(value, bool), (
                        f"Module {name}: incorrect value for terminal (should be a boolean)"
                    )
                    self._terminal = value
                    self._hard = True
                case _:
                    raise Exception(f"Module {name}: unknown module attribute")

        assert not self.is_hard or self.aspect_ratio is None, (
            f"Module {name}: aspect ratio incompatible with hard or fixed module"
        )

        assert not self.is_terminal or not self.is_fixed or self.center is not None, (
            f"Module {name}: a fixed terminal must have coordinates (center)."
        )

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
    def center(self) -> Optional[Point]:
        return self._center

    @center.setter
    def center(self, p: Optional[Point]) -> None:
        self._center = p

    # Getter and setter for min_shape
    @property
    def aspect_ratio(self) -> AspectRatio | None:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, ar: AspectRatio) -> None:
        self._aspect_ratio = ar

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @is_terminal.setter
    def is_terminal(self, value: bool) -> None:
        self._terminal = value

    @property
    def is_hard(self) -> bool:
        return self._hard

    @is_hard.setter
    def is_hard(self, value: bool) -> None:
        self._hard = value

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
        self._rectangles = list[Rectangle]()

    def add_rectangle(self, r: Rectangle) -> None:
        """
        Adds a rectangle to the module
        :param r: the rectangle
        """
        self.rectangles.append(r)
        self._area_rectangles = -1

    def recenter_rectangles(self) -> None:
        """Re-centers the rectangles of a hard module
        according to its center"""
        # Calculate current center
        assert self.is_hard and not self.is_fixed and self.center is not None
        area = sum(r.area for r in self.rectangles)
        x = sum(r.center.x * r.area for r in self.rectangles) / area
        y = sum(r.center.y * r.area for r in self.rectangles) / area
        inc_x, inc_y = self.center.x - x, self.center.y - y
        for r in self.rectangles:
            r.center.x += inc_x
            r.center.y += inc_y

    @staticmethod
    def _read_region_area(area: float | dict[str, float]) -> dict[str, float]:
        """Treats the area of a module as a float or as a dictionary of
        {region: float}, where region is a string
        :param area: area of the region. If only one number is specified,
        the area is assigned to the ground region.
        :return: a dictionary with the area associated to each region.
        """
        if isinstance(area, (int, float)):
            float_area = float(area)
            assert float_area > 0, "Area must be positive"
            return {KW.GROUND: float_area}

        dict_area: dict[str, float] = {}
        assert isinstance(area, dict), "Invalid area specification"
        for region, a in area.items():
            assert valid_identifier(region), "Invalid region identifier"
            assert is_number(a) and a > 0, "Invalid value for area"
            dict_area[region] = float(a)
        return dict_area

    def setup(self) -> None:
        """
        Checking the consistency of the module. No area must have been defined
        for hard/fixed modules. For soft blocks, the area must have been
        defined. The rectangles for hard modules must not overlap.
        The first rectangle of hard blocks must be the trunk
        """

        assert not (self.is_fixed and not self.is_hard), (
            f"Inconsistent fixed module {self.name}. It should be also hard."
        )

        assert not (self.flip and self.is_fixed), (
            f"Fixed module {self.name} cannot be flipped."
        )
        assert not (self.flip and not self.is_hard), (
            f"Soft module {self.name} cannot be flipped."
        )

        area_defined = len(self.area_regions) > 0
        assert self.is_hard or area_defined, (
            f"No area defined for a soft module {self.name}."
        )

        # terminal implies hard
        assert not self.is_terminal or self.is_hard, (
            f"Terminal module {self.name} is not hard."
        )

        if self.is_hard:
            # Check that neither area, nor center nor min_shape are defined.
            # It also checks that at least has one rectangle
            assert not area_defined, (
                f"Inconsistent hard module {self.name}: cannot specify area."
            )
            assert self.center is None or self.is_terminal, (
                f"Inconsistent hard module {self.name}: cannot specify center."
            )
            assert self.aspect_ratio is None, (
                f"Inconsistent hard module {self.name}: cannot specify aspect ratio."
            )
            assert self.is_terminal or self.num_rectangles > 0, (
                f"Inconsistent hard module {self.name}: must have at least "
                f"one rectangle or be a terminal."
            )

            # Calculate the area of hard modules
            area = sum(r.area for r in self.rectangles)
            self._area_regions = {KW.GROUND: area}
            self._total_area = area

    @property
    def has_strop(self) -> bool:
        """
        Determines whether a module is a STROP
        :return: True if it has an associate STROP, and False otherwise
        """
        return (
            self.num_rectangles > 0
            and self.rectangles[0].location == Rectangle.StropLocation.TRUNK
        )

    def create_square(self) -> None:
        """
        Creates a square for the module with the total area
        (and removes the previous rectangles)
        """
        assert self.center is not None, (
            f"Cannot calculate square for module {self.name}. Missing center."
        )
        area = self.area()
        assert area >= 0, (
            f"Cannot calculate square for module {self.name}. Area is zero."
        )
        side = math.sqrt(area)
        self._rectangles = list[Rectangle]()
        self.add_rectangle(
            Rectangle(**{KW.CENTER: self.center, KW.SHAPE: Shape(side, side)})
        )

    def create_strop(self) -> bool:
        """
        Defines the locations of the rectangles in a Single-Trunk Orthogon
        (STROP). It returns true if the module can be represented as a STROP of
        the rectangles. It also defines the locations of the rectangles in the
        STROP. If no polygon is found, all rectangles have the NO_POLYGON
        location
        :return: True if a STROP has been identified, and False otherwise
        """
        return create_strop(self.rectangles)

    def calculate_center_from_rectangles(self) -> Point:
        """
        Calculates the center from the rectangles.
        It raises an exception in case there are no rectangles.
        :return: the center of the module .
        """
        assert self.num_rectangles > 0, f"No rectangles in module {self.name}"
        sum_x, sum_y, area = 0.0, 0.0, 0.0
        for r in self.rectangles:
            sum_x += r.area * r.center.x
            sum_y += r.area * r.center.y
            area += r.area
        self.center = Point(sum_x / area, sum_y / area)
        assert isinstance(self.center, Point)  # just for type checking
        return self.center

    def __str__(self) -> str:
        s = f"{self.name}: {KW.AREA}={self.area_regions} {KW.CENTER}={self.center}"
        s += f" {KW.ASPECT_RATIO}={self.aspect_ratio}"
        if self.num_rectangles == 0:
            return s
        s += f" {KW.RECTANGLES}=["
        for r in self.rectangles:
            s += f"({str(r)})"
        s += "]"
        return s
