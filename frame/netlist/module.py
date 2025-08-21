# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Modules of a netlist
"""

import math
from typing import Optional, Any
from frame.geometry.geometry import Point, Shape, AspectRatio, Rectangle, create_strop
from frame.utils.keywords import KW
from frame.utils.utils import valid_identifier, is_number


class Module:
    """
    Class to represent a module of the system
    """

    _name: str  # Name of the module
    _center: Optional[Point]  # Center of the module (if defined)
    _aspect_ratio: Optional[AspectRatio]  # interval of the aspect ratio
    _iopin: bool  # It is an IO pin?
    _hard: bool  # Must be a hard module (but movable if not fixed)
    _fixed: bool  # Must be fixed in the layout
    _flip: bool  # May be flipped (only for hard modules, not fixed)
    # A module can be assigned to different layout areas (LUT, DSP, BRAM).
    # The following attribute indicate the area assigned to each region.
    # If no region specified, the default is assigned (Ground).
    _area_regions: dict[str, float]  # Area for each type of region
    # This is an attribute to store the whole area of the module.
    # If not calculated, it has a negative value.
    _total_area: float
    # A module can be implemented with different rectangles.
    # Here is the list of rectangles
    _rectangles: list[Rectangle]  # Rectangles of the module (if defined)
    # Total area of the rectangles (negative if not calculated).
    _area_rectangles: float
    _pin_length: float  # Length of the IO pin (negative if not defined)

    def __init__(self, name: str, **kwargs: dict[str, Any]):
        """
        Constructor
        :param kwargs: name (str), center (Point), aspect_ratio (AspectRatio),
                       area (float or dict),
        hard (boolean), fixed (boolean), IO pin (boolean)
        """
        self._name = name
        self._center = None
        self._aspect_ratio = None
        self._iopin = False
        self._hard = False
        self._fixed = False
        self._flip = False
        self._area_regions = {}
        self._total_area = -1.0
        self._rectangles = list[Rectangle]()
        self._area_rectangles = -1.0
        self._pin_length = -1.0

        self._read_parameters(name, kwargs)
        self._check_consistency(kwargs)

    def _read_parameters(self, name: str, kwargs: dict[str, Any]) -> None:
        """Parsers the information of a module"""

        # Check the name
        assert valid_identifier(name), "Incorrect module name"

        # Check that all boolean attributes, if present, are true (no false values allowed)
        for key, value in kwargs.items():
            if key in [KW.FIXED, KW.HARD, KW.FLIP, KW.IO_PIN]:
                assert isinstance(value, bool) and value, (
                    f"Module {name}: {key} must be true"
                )

        # Reading parameters and type checking
        for key, value in kwargs.items():
            match key:
                case KW.CENTER:
                    assert isinstance(value, Point), (
                        f"Module {name}: incorrect point associated to the center"
                    )
                    self._center = value
                case KW.ASPECT_RATIO:
                    assert (
                        isinstance(value, AspectRatio)
                        and 0 <= value.min_wh <= 1.0
                        and value.max_wh >= 1.0
                    ), f"Module {name}: incorrect aspect ratio"
                    self._aspect_ratio = value
                case KW.AREA:
                    self._area_regions = self._read_region_area(value)
                case KW.LENGTH:
                    assert isinstance(value, (int, float)) and value >= 0, (
                        f"Module {name}: incorrect value for length (should be a non-negative number)"
                    )
                    self._pin_length = float(value)
                case KW.FIXED:
                    self._fixed = self._hard = value
                case KW.HARD:
                    assert KW.FIXED not in kwargs, (
                        f"Module {name}: {KW.FIXED} and {KW.HARD} are mutually exclusive"
                    )
                    self._hard = value
                case KW.FLIP:
                    self._flip = value
                case KW.IO_PIN:
                    assert KW.AREA not in kwargs, (
                        f"Module {name}: IO pin must have length"
                    )
                    assert KW.ASPECT_RATIO not in kwargs, (
                        f"Module {name}: IO pin cannot have aspect ratio"
                    )
                    assert KW.FLIP not in kwargs, (
                        f"Module {name}: IO pin cannot have flip attribute"
                    )
                    self._iopin = value
                case KW.RECTANGLES:
                    for r in value:
                        assert isinstance(r, Rectangle), (
                            f"Module {name}: incorrect rectangle {r}"
                        )
                        self.add_rectangle(r)
                case _:
                    raise Exception(f"Module {name}: unknown module attribute")

    def _check_consistency(self, kwargs: dict[str, Any]) -> None:
        """Checks the consistency of the module"""

        modtype = "IO pin" if self.is_iopin else "module"

        # Check for a lot of inconsistencies

        # Blocks should have either center or rectangles, but not both
        assert self.num_rectangles == 0 or self.center is None, (
            f"Inconsistent {modtype} {self.name}. It cannot have both center and rectangles."
        )

        # calculate center from rectangles
        if self.num_rectangles > 0:
            self.calculate_center_from_rectangles()

        # Check that hard blocks do not have area/length, aspect ratio and center
        # They should only have rectangles
        if self.is_hard:
            for key in kwargs:
                if key in [KW.AREA, KW.LENGTH, KW.CENTER, KW.ASPECT_RATIO]:
                    raise Exception(
                        f"Inconsistent hard {modtype} {self.name}. It cannot define {key}."
                    )
            assert KW.RECTANGLES in kwargs, (
                f"Inconsistent hard {modtype} {self.name}. It must have at least one rectangle."
            )

        if not self.is_iopin:
            # If it is not an IO pin, it cannot have length
            assert self._pin_length < 0, (
                f"Inconsistent module {self.name}. It cannot define length."
            )

        if self.is_fixed:
            # Fixed ==> Cannot be flipped
            assert not self.flip, (
                f"Inconsistent fixed module {self.name}. It cannot be flipped."
            )

        if self.flip:
            # Only non-fixed hard modules can be flipped
            assert self.is_hard and not self.is_fixed, (
                f"Inconsistent flipped module {self.name}. It must be a hard and non-fixed module."
            )

        if self.is_hard:
            # Calculate the area of hard modules or length of IO pins
            if self.is_iopin:
                for r in self.rectangles:
                    assert r.is_line, (
                        f"IO pin {self.name} must be a line, but rectangle {r} is not."
                    )
                self._pin_length = sum(r.length for r in self.rectangles)
            else:
                area = sum(r.area for r in self.rectangles)
                self._area_regions = {KW.GROUND: area}
                self._total_area = area

        # Soft modules must define area or length
        area_length_defined = (
            self.pin_length >= 0 if self.is_iopin else len(self.area_regions) > 0
        )
        assert self.is_hard or area_length_defined, (
            f"No area/length defined for soft {modtype} {self.name}."
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
        """Returns the center of the module"""
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
    def is_iopin(self) -> bool:
        return self._iopin

    @is_iopin.setter
    def is_iopin(self, value: bool) -> None:
        self._iopin = value

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
    def area(self, region: Optional[str] = None) -> float:
        """Returns the area of a module associated to a region.
        If no region is specified, the total area is returned."""
        assert not self.is_iopin, f"Module {self.name} is an IO pin, so it has no area"
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
    def pin_length(self) -> float:
        """Returns the length of the IO pin (if it is an IO pin)"""
        assert self.is_iopin, (
            f"Module {self.name} is not an IO pin, so it has no length"
        )
        return self._pin_length

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
        (and removes the previous rectangles). For IO pins, it creates a point.
        """
        assert self.center is not None, (
            f"Cannot calculate square for module {self.name}. Missing center."
        )

        if self.is_iopin:
            # For IO pins, it creates a point
            self._rectangles = [
                Rectangle(**{KW.CENTER: self.center, KW.SHAPE: Shape(0, 0)})
            ]
            return
        area = self.area()
        assert area >= 0, (
            f"Cannot calculate square for module {self.name}. Area is zero."
        )
        side = math.sqrt(area)
        self._rectangles = [
            Rectangle(**{KW.CENTER: self.center, KW.SHAPE: Shape(side, side)})
        ]

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
        :return: the center of the module.
        """
        assert self.num_rectangles > 0, f"No rectangles in module {self.name}"
        sum_x, sum_y, total_area = 0.0, 0.0, 0.0
        # Special case: just one rectangle (possibly a pin with zero area)
        if self.num_rectangles == 1:
            self.center = self.rectangles[0].center
            return self.center

        # General case: multiple rectangles
        for r in self.rectangles:
            area_length = r.length if r.is_line else r.area
            sum_x += area_length * r.center.x
            sum_y += area_length * r.center.y
            total_area += area_length

        self.center = Point(sum_x / total_area, sum_y / total_area)
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
