"""
Module to represent netlists
"""

from io import StringIO
from typing import TextIO
from ruamel.yaml import YAML

from .module import Module
from .netlist_types import HyperEdge
from .yaml_read_netlist import parse_yaml_netlist
from .yaml_write_netlist import dump_yaml_modules, dump_yaml_edges
from ..geometry.geometry import Rectangle
from ..utils.keywords import KW_MODULES, KW_NETS


class Netlist:
    """
    Class to represent a netlist
    """

    _modules: list[Module]  # List of modules
    _edges: list[HyperEdge]  # List of edges, with references to modules
    _rectangles: list[Rectangle]  # List of rectangles
    _name2module: dict[str, Module]  # Map from module names to modules

    def __init__(self, stream: str | TextIO):
        """
        Constructor of a netlist from a file or from a string of text
        :param stream: name of the YAML file (str) or handle to the file
        """

        self._modules, _named_edges = parse_yaml_netlist(stream)
        self._name2module = {b.name: b for b in self._modules}

        # Edges
        self._edges = []
        for e in _named_edges:
            modules = []
            for b in e.modules:
                assert b in self._name2module, f'Unknown module {b} in edge'
                modules.append(self._name2module[b])
            assert e.weight > 0, f'Incorrect edge weight {e.weight}'
            self._edges.append(HyperEdge(modules, e.weight))

        # Create rectangles
        self._rectangles = [r for b in self.modules for r in b.rectangles]

    @property
    def num_modules(self) -> int:
        """Number of modules of the netlist"""
        return len(self._modules)

    @property
    def modules(self) -> list[Module]:
        """List of modules of the netlist"""
        return self._modules

    @property
    def edges(self) -> list[HyperEdge]:
        """List of hyperedges of the netlist"""
        return self._edges

    @property
    def num_rectangles(self) -> int:
        """Number of rectangles of all modules of the netlist"""
        return len(self.rectangles)

    @property
    def rectangles(self) -> list[Rectangle]:
        """Rectangles of all modules of the netlist"""
        return self._rectangles

    def create_squares(self) -> list[Module]:
        """
        Creates a default rectangle (square) for each module that has no rectangles
        :return: The list of modules for which a square has been created.
        """
        modules = []
        for b in self.modules:
            if b.num_rectangles == 0:
                b.create_square()
                modules.append(b)
        return modules

    def dump_yaml_netlist(self, filename: str = None) -> None | str:
        """
        Writes the netlist into a YAML file. If no file name is given, a string with the yaml contents
        is returned
        :param filename: name of the output file
        """
        data = {
            KW_MODULES: dump_yaml_modules(self.modules),
            KW_NETS: dump_yaml_edges(self.edges)
        }

        yaml = YAML()
        yaml.default_flow_style = False
        if filename is None:
            string_stream = StringIO()
            yaml.dump(data, string_stream)
            output_str: str = string_stream.getvalue()
            string_stream.close()
            return output_str
        with open(filename, 'w') as stream:
            yaml.dump(data, stream)
