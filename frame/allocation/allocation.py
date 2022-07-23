from typing import NamedTuple
from ..geometry.geometry import Point, Shape, Rectangle
from ..utils.utils import TextIO_String, read_yaml, write_yaml, YAML_tree, is_number, valid_identifier
from ..utils.keywords import KW_CENTER, KW_SHAPE

Alloc = dict[str, float]  # Allocation in a rectangle (area ratio for each module)


# Representation of the allocation in a rectangle.
class RectAlloc(NamedTuple):
    rect: Rectangle  # Rectangle of the allocation
    alloc: Alloc  # Area ratio for each module (in [0,1])


# Representation of allocation in a rectangle.
class ModuleAlloc(NamedTuple):
    rect: int  # Rectangle index of the allocation
    area: float  # Area ratio (in [0,1])


class Allocation:
    """Class to represent the allocation of modules into die rectangles.
    An allocation is represented by a set of rectangles. Each rectangle is occupied by a set
    of modules. The occupation is represented as a ratio, e.g., 10% of a rectangle is occupied by M1,
    30% by M2, etc. Ratios are represented as numbers in the interval [0,1]."""

    _allocations: list[RectAlloc]  # List of allocations. Each component corresponds to a rectangle
    _module2rect: dict[str, list[ModuleAlloc]]  # For each module, a list of rectangle allocations

    def __init__(self, stream: TextIO_String):
        """
        Redas a YAML specification of the allocation of rectangles.
        :param stream: It can be a name file, or a YAML specifcation (in text). The constructor can figure out
        which one it is.
        """
        self._parse_yaml_tree(read_yaml(stream))

    @property
    def num_rectangles(self) -> int:
        """
        :return: the number of rectangles of the allocation
        """
        return len(self._allocations)

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

    def _parse_yaml_tree(self, tree: YAML_tree) -> None:
        """"""
        assert isinstance(tree, list), "Wrong format of the allocation. The top node should be a list."
        self._allocations = []
        self._module2rect = {}
        for i, alloc in enumerate(tree):
            # Read one rectangle allocation
            assert isinstance(alloc, list) and len(alloc) == 2, f"Incorrect format for rectangle %d" % (i + 1)
            r, d = alloc[0], alloc[1]  # Rectangle and dictionary of allocations

            # Create the rectangle
            assert isinstance(r, list) and len(r) == 4, f"Incorrect format for rectangle %d" % (i + 1)
            for j in range(4):
                assert is_number(r[j]) and r[j] >= 0, f"Incorrect value for rectangle %d" % (i + 1)
            kwargs = {KW_CENTER: Point(r[0], r[1]), KW_SHAPE: Shape(r[2], r[3])}
            rect = Rectangle(**kwargs)

            # Create the dictionary of allocations
            assert isinstance(d, dict), f"Incorrect allocation for rectangle %d" % (i + 1)
            dict_alloc = {}
            total_occup = 0
            for module, occup in d.items():
                assert valid_identifier(module), f"Invalid module identifier: {module}"
                assert 0 < occup <= 1, f"Invalid allocation for {module} in redtangle %d" % (i + 1)
                assert module not in dict_alloc, f"Multiple allocations of {module} in rectangle %d" % (i + 1)
                total_occup += occup
                dict_alloc[module] = occup
                if module not in self._module2rect:
                    self._module2rect[module] = []
                self._module2rect[module].append(ModuleAlloc(i, occup))

            assert total_occup <= 1.0, f"Occupation of rectangle %d greater than 1" % (i + 1)
            self._allocations.append(RectAlloc(rect, dict_alloc))

    def write_yaml(self, filename: str = None) -> None | str:
        """
        Writes the allocation into a YAML file. If no file name is given, a string with the yaml contents
        is returned
        :param filename: name of the output file
        """
        list_modules = [[r.rect.vector_spec, r.alloc] for r in self._allocations]
        write_yaml(list_modules, filename)