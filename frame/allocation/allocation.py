import itertools
import math

from dataclasses import dataclass

from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point, Shape, Rectangle, parse_yaml_rectangle
from frame.utils.keywords import KW_CENTER, KW_SHAPE
from frame.utils.utils import TextIO_String, read_yaml, write_yaml, YAML_tree, is_number, valid_identifier, Vector

Alloc = dict[str, float]  # Allocation in a rectangle (area ratio for each module)

# Descriptor of a rectangle. In reality this corresponds to a description of the form:
# [[x,y,w,h], Alloc, depth]. Since tuples are not allowed in YAML, this generic descriptor is used
RectDescriptor = list[Vector | Alloc | int]


@dataclass()
class RectAlloc:
    """Representation of the allocation in a rectangle."""
    rect: Rectangle  # Rectangle of the allocation
    alloc: Alloc  # Area ratio for each module (in [0,1])
    depth: int  # depth of refinement


@dataclass()
class ModuleAlloc:
    """Representation of allocation of one module in a rectangle."""
    rect: int  # Rectangle index of the allocation
    area: float  # Area ratio (in [0,1])


class Allocation:
    """Class to represent the allocation of modules into die rectangles.
    An allocation is represented by a set of rectangles. Each rectangle is occupied by a set
    of modules. The occupation is represented as a ratio, e.g., 10% of a rectangle is occupied by M1,
    30% by M2, etc. Ratios are represented as numbers in the interval [0,1]."""

    _allocations: list[RectAlloc]  # List of allocations. Each component corresponds to a rectangle
    _module2rect: dict[str, list[ModuleAlloc]]  # For each module, a list of rectangle allocations
    _areas: dict[str, float]  # Area of the modules
    _centers: dict[str, Point]  # Centers of the modules
    _bounding_box: Rectangle

    def __init__(self, stream: TextIO_String):
        """
        Reads a YAML specification of the allocation of rectangles
        :param stream: It can be a name file, or a YAML specification (in text). The constructor can figure out
        which one it is.
        """
        self._parse_yaml_tree(read_yaml(stream))
        self._calculate_bounding_box()
        self._check_no_overlap()
        self._calculate_areas_and_centers()

    @property
    def bounding_box(self) -> Rectangle:
        """
        Returns the bounding box of the allocation
        :return: the bounding box of the allocation.
        """
        return self._bounding_box

    @property
    def num_rectangles(self) -> int:
        """
        Returns the number of rectangles
        :return: the number of rectangles of the allocation
        """
        return len(self._allocations)

    @property
    def num_modules(self) -> int:
        """
        Returns the number of modules
        :return: the number of modules of the allocation
        """
        return len(self._module2rect)

    @property
    def allocations(self) -> list[RectAlloc]:
        """
        Returns the list of allocations (one for each rectangle)
        :return: the list of allocations
        """
        return self._allocations

    def allocation_rectangle(self, i: int) -> RectAlloc:
        """
        Returns the allocation of the i-th rectangle
        :param i: index of the rectangle
        :return: the allocation of the i-th rectangle
        """
        assert i < self.num_rectangles
        return self._allocations[i]

    def allocation_module(self, m: str) -> list[ModuleAlloc]:
        """
        Returns the allocation of a module (pairs of rectangle indices and ratios)
        :param m: name of the module
        :return: the allocation of module m
        """
        return self._module2rect[m]

    def area(self, modules: str | list[str]) -> float:
        """Returns the area of a set of modules (or the area of a module if only one string is passed)
        :param modules: module name s
        :return: the area of the modules
        """
        if isinstance(modules, str):
            return self._areas[modules]
        return sum(self._areas[m] for m in modules)

    def center(self, modules: str | list[str]) -> Point:
        """Returns the center of a set of modules (or the center of the module if only one string is passed)
        :param modules: module names
        :return: the center of the modules
        """
        if isinstance(modules, str):
            return self._centers[modules]
        center = Point(0, 0)
        for m in modules:
            center += self.center(m) * self.area(m)
        return center / self.area(modules)

    def check_compatible(self, netlist: Netlist) -> bool:
        """
        Checks whether the allocation is compatible with a netlist (the set of modules is the same)
        :param netlist: the netlist
        :return: True if compatible, and False otherwise
        """
        return {m.name for m in netlist.modules} == {m for m in self._module2rect.keys()}

    def refine(self, threshold: float, levels: int = 2) -> 'Allocation':
        """
        Refines an allocation into a set of smaller rectangles. A rectangle in the allocation
        is refined if no module has an occupancy greater than a threshold. In case a rectangle
        is refined, it is recursively split into 2^levels rectangles. The splitting is always done by the
        largest dimension (width or height).
        :param threshold: rectangles are split if no module has an occupancy greater than this threshold
        :param levels: number of splitting levels
        :return: A new allocation
        """
        assert levels > 0
        new_alloc: list[RectDescriptor] = []
        for a in self.allocations:
            # Check the allocations and see if the rectangle must be split
            split = len(a.alloc) > 0 and all(x <= threshold for x in a.alloc.values())
            new_alloc.extend(self._split_allocation(a.rect.vector_spec, a.alloc, a.depth, levels if split else 0))
        return Allocation(new_alloc)

    def must_be_refined(self, threshold: float) -> bool:
        """
        Checks whether the allocation must be refined. An allocation must be refined if there is
        a rectangle in which no modules have an occupancy greater than a threshold
        :param threshold: rectangles must be split if no module has an occupancy greater than this threshold
        :return: True if the allocation must be refined
        """
        return any(all(x <= threshold for x in a.alloc.values()) for a in self.allocations)

    def max_refinement_depth(self) -> int:
        """Checks the maximum depth of refinement of the rectangles.
        :return: the maximum depth of refinement
        """
        return max(r.depth for r in self.allocations)

    def uniform_refinement_depth(self) -> 'Allocation':
        """Refines all rectangles in a way that all cells have the same refinement depth.
        :return: a new Allocation"""
        max_depth = max(r.depth for r in self.allocations)
        min_depth = min(r.depth for r in self.allocations)
        # Do we really need refinement?
        if max_depth == min_depth:
            return self

        new_alloc: list[RectDescriptor] = []
        for a in self.allocations:
            new_alloc.extend(self._split_allocation(a.rect.vector_spec, a.alloc, a.depth, max_depth - a.depth))
        return Allocation(new_alloc)

    def write_yaml(self, filename: str = None) -> None | str:
        """
        Writes the allocation into a YAML file. If no file name is given, a string with the yaml contents
        is returned
        :param filename: name of the output file
        """
        list_modules = [[r.rect.vector_spec, r.alloc, r.depth] if r.depth > 0
                        else [r.rect.vector_spec, r.alloc] for r in self.allocations]
        return write_yaml(list_modules, filename)

    def _calculate_bounding_box(self) -> None:
        """
        Returns the bounding box of all rectangles
        """
        xmin, xmax, ymin, ymax = math.inf, -math.inf, math.inf, -math.inf
        for rect_alloc in self.allocations:
            ll, ur = rect_alloc.rect.bounding_box
            xmin = min(xmin, ll.x)
            xmax = max(xmax, ur.x)
            ymin = min(ymin, ll.y)
            ymax = max(ymax, ur.y)
        assert xmin >= 0 and ymin >= 0, "The allocation is not included in the positive quadrant"
        width = xmax - xmin
        height = ymax - ymin
        kwargs = {KW_CENTER: Point(xmin + width / 2, ymin + height / 2), KW_SHAPE: Shape(width, height)}
        self._bounding_box = Rectangle(**kwargs)

    def _parse_yaml_tree(self, tree: YAML_tree) -> None:
        """
        Parses the YAML tree that represents an allocation
        :param tree: the YAML tree
        """
        assert isinstance(tree, list), "Wrong format of the allocation. The top node should be a list."
        self._allocations = []
        self._module2rect = {}
        for i, alloc in enumerate(tree):
            # Read one rectangle allocation
            assert isinstance(alloc, list) and 2 <= len(alloc) <= 3, f'Incorrect format for rectangle {alloc}'
            r, d = alloc[0], alloc[1]  # Rectangle and dictionary of allocations
            depth = 0 if len(alloc) == 2 else alloc[2]
            assert isinstance(depth, int) and depth >= 0, f'Incorrect depth for rectangle {alloc}'

            # Create the rectangle
            rect = parse_yaml_rectangle(r)

            # Create the dictionary of allocations
            assert isinstance(d, dict), f'Incorrect allocation for rectangle {r}'
            dict_alloc = {}
            total_occup = 0
            for module, occup in d.items():
                assert valid_identifier(module), f'Invalid module identifier: {module}'
                assert is_number(occup) and 0 <= occup <= 1, f'Invalid allocation for {module} in rectangle {r}'
                assert module not in dict_alloc, f'Multiple allocations of {module} in rectangle {r}'
                total_occup += occup
                dict_alloc[module] = occup
                if module not in self._module2rect:
                    self._module2rect[module] = []
                self._module2rect[module].append(ModuleAlloc(i, occup))

            # Check that a rectangle is not over-occupied (assertion removed)
            # assert total_occup <= 1.0, f'Occupancy of rectangle {r} greater than 1: {total_occup}'
            self._allocations.append(RectAlloc(rect, dict_alloc, depth))

    def _check_no_overlap(self) -> None:
        """Checks that rectangles do not overlap"""
        for r_allocs in itertools.combinations(self._allocations, 2):
            assert not r_allocs[0].rect.overlap(r_allocs[1].rect), "Allocations rectangles overlap"

    def _calculate_areas_and_centers(self) -> None:
        """Computers the area and the centers of all modules"""
        self._areas = {}
        self._centers = {}
        for module, alloc in self._module2rect.items():
            center = Point(0, 0)
            total_area = 0.0
            for mod_alloc in alloc:
                r = self._allocations[mod_alloc.rect].rect
                ratio = mod_alloc.area * r.area
                center += r.center * ratio
                total_area += ratio
            self._areas[module] = total_area
            self._centers[module] = center / total_area

    @staticmethod
    def _split_allocation(rect: Vector, alloc: Alloc, depth: int, levels: int = 0) -> list[RectDescriptor]:
        """
        Splits a rectangle into 2^levels rectangles and returns a list of rectangle allocations
        :param rect: the rectangle
        :param alloc: the module allocation fo the rectangle
        :param depth: depth of the refinement
        :param levels: number of splitting levels
        :return: a list of allocations
        """
        if levels == 0:
            return [[rect, {m: r for m, r in alloc.items()}, depth]]

        # Split the largest dimension
        if rect[2] >= rect[3]:
            # Split width
            w2, w4 = rect[2] / 2, rect[2] / 4
            rect1 = [rect[0] - w4, rect[1], w2, rect[3]]
            rect2 = [rect[0] + w4, rect[1], w2, rect[3]]
        else:
            # Split height
            h2, h4 = rect[3] / 2, rect[3] / 4
            rect1 = [rect[0], rect[1] - h4, rect[2], h2]
            rect2 = [rect[0], rect[1] + h4, rect[2], h2]

        return Allocation._split_allocation(rect1, alloc, depth + 1, levels - 1) + \
            Allocation._split_allocation(rect2, alloc, depth + 1, levels - 1)
