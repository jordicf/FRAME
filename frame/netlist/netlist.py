# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Module to represent netlists
"""

import math
from itertools import combinations
from typing import Optional, Iterator
from frame.netlist.module import Module
from frame.netlist.netlist_types import HyperEdge
from frame.netlist.yaml_read_netlist import parse_yaml_netlist
from frame.netlist.yaml_write_netlist import dump_yaml_modules, dump_yaml_edges
from frame.geometry.geometry import Rectangle, parse_yaml_rectangle
from frame.utils.keywords import KW_MODULES, KW_NETS
from frame.utils.utils import write_json_yaml, Python_object

# Data structure to represent the rectangles associated to a module.
# For each module, a list of rectangles is defined.
# Each rectangle is a list of four values: [x, y, w, h].
# Optionally, a fifth value (string representing the regions) can be added,
# e.g., [3, 5, 2, 8.5, "dsp"]
RectangleRepr = list[float | str]
Module2Rectangles = dict[str, list[RectangleRepr]]


class Netlist:
    """
    Class to represent a netlist
    """

    _modules: list[Module]  # List of modules
    _edges: list[HyperEdge]  # List of edges, with references to modules
    _name2module: dict[str, Module]  # Map from module names to modules
    _rectangles: Optional[list[Rectangle]]  # List of rectangles

    def __init__(self, stream: str):
        """
        Constructor of a netlist from a file or from a string of text (YAML).
        The file can be either in JSON or YAML.
        :param stream: name of the file or the YAML string
        """

        self._modules, _named_edges = parse_yaml_netlist(stream)
        self._name2module = {b.name: b for b in self._modules}
        self._create_rectangles()

        # Edges
        self._edges = list[HyperEdge]()
        for e in _named_edges:
            modules = list[Module]()
            for b in e.modules:
                assert b in self._name2module, f"Unknown module {b} in edge"
                modules.append(self._name2module[b])
            assert e.weight > 0, f"Incorrect edge weight {e.weight}"
            self._edges.append(HyperEdge(modules, e.weight))

    @property
    def num_modules(self) -> int:
        """Number of modules of the netlist"""
        return len(self._modules)

    @property
    def modules(self) -> list[Module]:
        """List of modules of the netlist"""
        return self._modules

    @property
    def num_edges(self) -> int:
        """Number of hyperedges of the netlist"""
        return len(self._edges)

    @property
    def edges(self) -> list[HyperEdge]:
        """List of hyperedges of the netlist"""
        return self._edges

    @property
    def wire_length(self) -> float:
        """Total wire length to construct the netlist
        (sum of netlist hyperedges wire lengths)"""
        return sum([e.wire_length for e in self.edges])

    @property
    def num_rectangles(self) -> int:
        """Number of rectangles of all modules of the netlist"""
        return len(self.rectangles)

    @property
    def rectangles(self) -> list[Rectangle]:
        """Rectangles of all modules of the netlist"""
        if self._rectangles is None:
            self._create_rectangles()
        assert self._rectangles is not None
        return self._rectangles

    def get_module(self, name: str) -> Module:
        """
        Returns the module with a certain name
        :param name: name mof the module
        :return: the module
        """
        assert name in self._name2module, f"Module {name} does not exist"
        return self._name2module[name]

    def create_squares(self) -> list[Module]:
        """
        Creates a default rectangle (square) for each module
        that has no rectangles
        :return: The list of modules for which a square has been created.
        """
        modules = list[Module]()
        for m in self.modules:
            if m.num_rectangles == 0:
                m.create_square()
                self._clean_rectangles()
                modules.append(m)
        return modules

    def create_strops(self) -> None:
        """
        Creates the Single-Trunk Orthogonal Polygons (STROPs) for each module
        (if they can be identified as STROPs). The location of the rectangles
        of each module a labelled according to their role. If no STROP
        can be identified, the rectangles are labelled as NO_POLYGON
        """
        for m in self.modules:
            m.create_strop()

    def all_soft_modules_have_strops(self) -> bool:
        """Indicates whether all soft modules have STROPs"""
        return all(m.is_hard or m.has_strop for m in self.modules)

    def get_rectangles_module(self, module: str) -> list[Rectangle]:
        """Returns the list of rectangles of a module"""
        return self.get_module(module).rectangles

    def assign_rectangles_module(self, module: str, rects: Iterator[Rectangle]) -> None:
        """
        Defines the rectangles of a module of the netlist.
        The previous rectangles are removed.
        :param module: Name of the module
        :param rects: set of rectangles
        """
        m = self.get_module(module)
        m.clear_rectangles()
        for r in rects:
            m.add_rectangle(r)
        self._clean_rectangles()

    def assign_rectangles(self, mod_rect: dict[str, Iterator[Rectangle]]) -> None:
        """
        Defines the rectangles of a set modules of the netlist
        :param m2r: The rectangles associated to every module
        """
        for module, rects in mod_rect.items():
            self.assign_rectangles_module(module, rects)

    def fixed_rectangles(self) -> list[Rectangle]:
        """
        Returns the list of fixed rectangles
        :return: the list of fixed rectangles
        """
        return [r for r in self.rectangles if r.fixed]

    def write_yaml(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Writes the netlist into a YAML file.
        If no file name is given, a string with the yaml contents is returned
        :param filename: name of the output file
        """
        data = self._write_json_yaml_data()
        return write_json_yaml(data, False, filename)

    def write_json(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Writes the netlist into a JSON file. If no file name is given,
        a string with the JSON contents is returned
        :param filename: name of the output file
        """
        data = self._write_json_yaml_data()
        return write_json_yaml(data, True, filename)

    def _write_json_yaml_data(self) -> Python_object:
        """
        Generates the data structure to be dumped into a JSON or YAML file.
        """
        return {
            KW_MODULES: dump_yaml_modules(self.modules),
            KW_NETS: dump_yaml_edges(self.edges),
        }

    def _clean_rectangles(self) -> None:
        """
        Removes all the rectangles of the netlist
        """
        self._rectangles = None

    def _create_rectangles(self) -> None:
        """
        Creates the list of rectangles of the netlist. For hard nodes without
        rectangles, it creates a square. It also defines epsilon, in case it
        was not defined
        """
        self._clean_rectangles()
        # Check that all fixed nodes have either a center or a rectangle
        smallest_distance = math.inf
        for m in self.modules:
            assert (
                m.is_terminal
                or m.is_soft
                or m.center is not None
                or m.num_rectangles > 0
            ), f"Module {m.name} is hard and has neither center nor rectangles"
            if m.is_hard and not m.is_terminal and m.num_rectangles == 0:
                m.create_square()
            if m.num_rectangles > 0:
                m.calculate_center_from_rectangles()
        self._rectangles = [r for b in self.modules for r in b.rectangles]

        if not Rectangle.epsilon_defined():
            for r in self.rectangles:
                smallest_distance = min(smallest_distance, r.shape.w, r.shape.h)
            for m in self.modules:
                a = m.area()
                if a > 0:
                    smallest_distance = min(smallest_distance, math.sqrt(a))
            Rectangle.set_epsilon(smallest_distance * 1e-12)

        # Check that hard modules have non-overlapping rectangles.
        for m in self.modules:
            if m.is_hard and not m.is_terminal:
                for r1, r2 in combinations(m.rectangles, 2):
                    assert not r1.overlap(r2), (
                        f"Inconsistent hard module {m.name}: overlapping rectangles."
                    )

        # Create strops
        for m in self.modules:
            if m.num_rectangles > 0:
                m.create_strop()

        assert all(not m.flip or m.has_strop for m in self.modules), (
            "Not all flip modules have a STROP"
        )
