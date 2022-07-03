import math

from geometry import Point, Shape, Rectangle
from keywords import KW_CENTER, KW_SHAPE, KW_MIN_SHAPE, KW_AREA, KW_FIXED, KW_FAKE, KW_GROUND
from utils import valid_identifier, is_number


class Block:
    """
    Class to represent a block of the system
    """
    _name: str  # Name of the block
    _center: Point | None  # Center of the block (if defined)
    _min_shape: Shape | None  # min width and height
    _fixed: bool  # Must be fixed in the layout
    _fake: bool  # Used for fake nodes (centers of hyperedges)
    # A block can be assigned to different layout areas (LUT, DSP, BRAM).
    # The following attribute indicate the area assigned to each region.
    # If no region specified, the default is assigned (Ground).
    _area_regions: dict[str, float]  # Area for each type of region
    # This is an attribute to store the whole area of the block. If not calculated, it
    # has a negative value
    _total_area: float
    # A block can be implemented with different rectangles. Here is the list of rectangles
    _rectangles: list[Rectangle]  # Rectangles of the block (if defined)
    _area_rectangles: float # Total area of the rectangles (negative if not calculated).
    # Allocation of regions. This dictionary receives the name of a rectangle (from the die)
    # as key and stores the ratio of ocupation of the rectangle by the block (a value in [0,1]).
    _area_rectangles: float
    _alloc: dict[str, float]  # Allocation to regions (rectangles) defined in the die

    def __init__(self, name: str, **kwargs):
        """
        Constructor
        :param kwargs: name (str), center (Point), min_shape (Shape), area (float or dict), fixed (boolean)
        """
        self._name = name
        self._center = None
        self._min_shape = None
        self._fixed = False
        self._fake = False
        self._area_regions = {}
        self._total_area = -1
        self._rectangles = []
        self._area_rectangles = -1
        self._alloc = {}

        assert valid_identifier(name), "Incorrect block name"

        # Reading parameters and type checking
        for key, value in kwargs.items():
            assert key in [KW_CENTER, KW_MIN_SHAPE, KW_AREA, KW_FIXED, KW_FAKE], "Unknown block attribute"
            if key == KW_CENTER:
                assert isinstance(value, Point), "Incorrect point associated to the center of the block"
                self._center = value
            elif key == KW_MIN_SHAPE:
                assert isinstance(value, Shape), "Incorrect shape associated to the block"
                assert value.w >= 0, "Incorrect block width"
                assert value.h >= 0, "incorrect block height"
                self._min_shape = value
            elif key == KW_AREA:
                self._area_regions = self._read_region_area(value)
            elif key == KW_FIXED:
                assert isinstance(value, bool), "Incorrect value for fixed (should be a boolean)"
                self._fixed = value
            elif key == KW_FAKE:
                assert isinstance(value, bool), "Incorrect value for fake (should be a boolean)"
                self._fake = value
            else:
                assert False  # Should never happen

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: 'Block') -> bool:
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
    def fixed(self) -> bool:
        return self._fixed

    @property
    def fake(self) -> bool:
        return self._fake

    # Getters for area
    def area(self, region: str | None = None) -> float:
        """Returns the area of a block associated to a region.
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
        return len(self.rectangles)

    @property
    def rectangles(self) -> list[Rectangle]:
        return self._rectangles

    @property
    def area_rectangles(self) -> float:
        """Returns the total area of the rectangles of the block"""
        if self._area_rectangles < 0:
            self._area_rectangles = sum(r.area for r in self.rectangles)
        return self._area_rectangles

    def add_rectangle(self, r: Rectangle) -> None:
        """
        Adds a rectangle to the block
        :param r: the rectangle
        """
        self.rectangles.append(r)
        self._area_rectangles = -1

    def name_rectangles(self) -> None:
        """
        Defines the names of the rectangles (block[idx] or simply block)
        """
        if self.num_rectangles == 1:
            self.rectangles[0].name = self.name
            return

        for idx, r in enumerate(self.rectangles):
            r.name = self.name + f'[{idx}]'

    def add_allocation(self, region: str, usage: float) -> None:
        """
        Adds an allocation to a region
        :param region: name of the region where the block must be allocated (a rectangle in the layout)
        :param usage: usage of the region (ratio between 0 and 1)
        """
        assert 0 <= usage <= 1, "Invalid usage specification of a region"
        assert region not in self._alloc, f"Duplicated region allocation in block {self.name}: {region}"
        self._alloc[region] = usage
        self._total_area = -1

    @property
    def allocations(self) -> dict[str, float]:
        """
        :return: the set of allocations to regions of the layout
        """
        return self._alloc

    @staticmethod
    def _read_region_area(area: float | dict[str, float]) -> dict[str, float]:
        """Treats the area of a block as a float or as a dictionary of {region: float},
        where region is a string
        :param area: area of the region. If only one number is specified, the area is
        assigned to the ground region.
        :return: a dictionary with the area associated to each region.
        """
        if is_number(area):
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

    def create_square(self) -> None:
        """
        Creates a square for the block with the total area (and removes the previous rectangles)
        """
        assert self.center is not None, f"Cannot calculate square for block {self.name}. Missing center."
        area = self.area()
        assert area > 0, f"Cannot calculate square for block {self.name}. Area is zero."
        side = math.sqrt(area)
        assert self.center is not None, f"Undefined center for block {self.name}"
        self._rectangles = []
        self.add_rectangle(Rectangle(**{KW_CENTER: self.center, KW_SHAPE: Shape(side, side)}))

    def calculate_center_from_rectangles(self) -> None:
        """
        Calculates the center from the rectangles. It raises an exception in case there are no rectangles.
        :return: the center of the block .
        """
        assert self.num_rectangles > 0, f"No rectangles in block {self.name}"
        sum_x, sum_y, area = 0, 0, 0
        for r in self.rectangles:
            sum_x += r.area * r.center.x
            sum_y += r.area * r.center.y
            area += r.area
        self.center = Point(sum_x / area, sum_y / area)

    def __str__(self) -> str:
        s = f"{self.name}: {KW_AREA}={self.area_regions} {KW_CENTER}={self.center}"
        s += f" {KW_MIN_SHAPE}={self.min_shape}"
        if self.num_rectangles == 0:
            return s
        s += f" KW_RECTANGLES=["
        for r in self.rectangles:
            s += f"({str(r)})"
        s += "]"
        return s
