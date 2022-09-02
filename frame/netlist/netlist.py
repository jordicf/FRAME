"""
Module to represent netlists
"""

import math
from frame.netlist.module import Module
from frame.netlist.netlist_types import HyperEdge
from frame.netlist.yaml_read_netlist import parse_yaml_netlist
from frame.netlist.yaml_write_netlist import dump_yaml_modules, dump_yaml_edges
from frame.geometry.geometry import Rectangle, parse_yaml_rectangle
from frame.utils.keywords import KW_MODULES, KW_NETS
from frame.utils.utils import TextIO_String, write_yaml

# Data structure to represent the rectangles associated to a module.
# For each module, a list of rectangles is defined.
# Each rectangle is a list of four values: [x, y, w, h].
# Optionally, a fifth value (string representing the regions) can be added, e.g., [3, 5, 2, 8.5, "dsp"]
Module2Rectangles = dict[str, list[list[float | str]]]


class Netlist:
    """
    Class to represent a netlist
    """

    _modules: list[Module]  # List of modules
    _edges: list[HyperEdge]  # List of edges, with references to modules
    _name2module: dict[str, Module]  # Map from module names to modules
    _rectangles: list[Rectangle] | None  # List of rectangles

    def __init__(self, stream: TextIO_String):
        """
        Constructor of a netlist from a file or from a string of text
        :param stream: name of the YAML file (str) or handle to the file
        """

        self._modules, _named_edges = parse_yaml_netlist(stream)
        self._name2module = {b.name: b for b in self._modules}
        self._rectangles = None

        # Edges
        self._edges = []
        for e in _named_edges:
            modules: list[Module] = []
            for b in e.modules:
                assert b in self._name2module, f'Unknown module {b} in edge'
                modules.append(self._name2module[b])
            assert e.weight > 0, f'Incorrect edge weight {e.weight}'
            self._edges.append(HyperEdge(modules, e.weight))

        self._create_rectangles()

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
        """Total wire length to construct the netlist (sum of netlist hyperedges wire lengths)"""
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
        assert name in self._name2module, f'Module {name} does not exist'
        return self._name2module[name]

    def create_squares(self) -> list[Module]:
        """
        Creates a default rectangle (square) for each module that has no rectangles
        :return: The list of modules for which a square has been created.
        """
        modules = []
        for m in self.modules:
            if m.num_rectangles == 0:
                m.create_square()
                self._clean_rectangles()
                modules.append(m)
        return modules

    def create_stogs(self) -> None:
        """
        Creates the Single-Trunk Orthogons (STOGs) for each module (if they can be identified as STOGs).
        The location of the rectangles of each module a labelled according to their role. If no STOG
        can be identified, the rectangles are labelled as NO_POLYGON
        """
        for m in self.modules:
            m.create_stog()

    def all_soft_modules_have_stogs(self) -> bool:
        """Indicates whether all soft modules have Single-Trunk Orthogons"""
        return all(m.is_fixed or m.has_stog for m in self.modules)

    def assign_rectangles(self, m2r: Module2Rectangles) -> None:
        """
        Defines the rectangles of the modules of the netlist
        :param m2r: The rectangles associated to every module
        """
        for module_name, list_rect in m2r.items():
            m = self.get_module(module_name)
            m.clear_rectangles()
            for r in list_rect:
                m.add_rectangle(parse_yaml_rectangle(r, m.is_fixed, m.is_hard))
        self._clean_rectangles()

    def fixed_rectangles(self) -> list[Rectangle]:
        """
        Returns the list of fixed rectangles
        :return: the list of fixed rectangles
        """
        return [r for r in self.rectangles if r.fixed]

    def write_yaml(self, filename: str = None) -> None | str:
        """
        Writes the netlist into a YAML file. If no file name is given, a string with the yaml contents
        is returned
        :param filename: name of the output file
        """
        data = {
            KW_MODULES: dump_yaml_modules(self.modules),
            KW_NETS: dump_yaml_edges(self.edges)
        }
        return write_yaml(data, filename)

    def _clean_rectangles(self) -> None:
        """
        Removes all the rectangles of the netlist
        """
        self._rectangles = None

    def _create_rectangles(self) -> None:
        """
        Creates the list of rectangles of the netlist. For fixed nodes without rectangles,
        it creates a square. It also defined epsilon, in case it was not defined
        """
        # Check that all fixed nodes have either a center or a rectangle
        smallest_distance = math.inf
        for m in self.modules:
            assert not m.is_fixed or m.center is not None or m.num_rectangles > 0,\
                f'Module {m.name} is fixed and has neither center nor rectangles'
            if m.is_fixed and m.num_rectangles == 0:
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
            Rectangle.set_epsilon(smallest_distance*1e-12)

        for m in self.modules:
            if m.num_rectangles > 0:
                m.create_stog()
