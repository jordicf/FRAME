# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Module to read netlists in JSON/YAML format
"""

from typing import Any, cast
from frame.netlist.module import Module
from frame.netlist.netlist_types import NamedHyperEdge
from frame.geometry.geometry import Rectangle, AspectRatio, Point, parse_yaml_rectangle
from frame.utils.keywords import KW_RECTANGLES, KW_CENTER, KW_TERMINAL, KW_FIXED, KW_HARD, KW_FLIP, \
    KW_MODULES, KW_NETS, KW_AREA, KW_ASPECT_RATIO
from frame.utils.utils import valid_identifier, is_number, read_json_yaml, TextIO_String


def parse_yaml_netlist(stream: TextIO_String) -> tuple[list[Module], list[NamedHyperEdge]]:
    """
    Parses a YAML netlist from a file or from a string of text. If the text has only one word, it is assumed
    to be a file name
    :param stream: name of the YAML file, YAML text or handle to the file
    :return: the list of modules and the list of edges.
    """
    tree = read_json_yaml(stream)
    assert isinstance(tree, dict), "The YAML root node is not a dictionary"
    modules: list[Module] = []
    edges: list[NamedHyperEdge] = []
    for key, value in tree.items():
        assert key in [KW_MODULES, KW_NETS], f"Unknown key {key}"
        if key == KW_MODULES:
            modules = parse_yaml_modules(value)
        elif key == KW_NETS:
            edges = parse_yaml_edges(value)
        else:
            assert False  # Should never happen
    return modules, edges


def parse_yaml_modules(modules: dict) -> list[Module]:
    """
    Parses the modules of the netlist
    :param modules: The collection of modules
    :return: the list of modules
    """
    assert isinstance(
        modules, dict), "The YAML node for modules is not a dictionary"
    _modules: list[Module] = []
    for name, module_info in modules.items():
        assert valid_identifier(name), f"Invalid module name: {name}"
        _modules.append(parse_yaml_module(name, module_info))
    return _modules


def parse_yaml_module(name: str, info: dict) -> Module:
    """
    Parses the information of a module
    :param name: Name of the module
    :param info: Information of the module
    :return: a module
    """
    assert isinstance(
        info, dict), f"The YAML node for module {name} is not a dictionary"
    assert valid_identifier(name), f"Invalid name for module: {name}"

    params: dict[str, Any] = {}
    for key, value in info.items():
        assert isinstance(key, str)
        if key in [KW_AREA, KW_TERMINAL, KW_FIXED, KW_HARD, KW_FLIP]:
            params[key] = value
        elif key == KW_CENTER:
            params[KW_CENTER] = parse_yaml_center(value, name)
        elif key == KW_ASPECT_RATIO:
            params[KW_ASPECT_RATIO] = parse_yaml_aspect_ratio(value, name)
        elif key == KW_RECTANGLES:
            pass
        else:
            assert False, f"Unknown module attribute {key}"

    m = Module(name, **params)

    if KW_RECTANGLES in info:
        rectangles = parse_yaml_rectangles(
            info[KW_RECTANGLES], m.is_fixed, m.is_hard)
        for r in rectangles:
            m.add_rectangle(r)

    m.setup()
    return m


def parse_yaml_center(center: list[float], name: str) -> Point:
    """
    Parses the center of a module
    :param center: module center [x, y]
    :param name: name of the block
    :return: a Point with (x,y)
    """
    assert isinstance(center, list) and len(center) == 2 and is_number(center[0]) and is_number(center[1]), \
        f"Incorrect format for the center of module {name}"
    return Point(float(center[0]), float(center[1]))


def parse_yaml_aspect_ratio(aspect_ratio: float | list[float], name: str) -> AspectRatio:
    """
    Parses the aspect ratio of the module. If only one value is given, the aspect ratio is computed
    as the interval [value, 1/value] or [1/value, value] in such a way that the first component is smaller
    than the second
    :param aspect_ratio: module attribute
    :param name: name of the module
    :return: an AspectRatio with (min w/h, max w/h)
    """
    if is_number(aspect_ratio):
        ar = cast(float, aspect_ratio)
        assert ar > 0, f"Incorrect aspect ratio for module {name}"
        inv_ar = 1 / ar
        return AspectRatio(float(min(ar, inv_ar)), float(max(ar, inv_ar)))

    assert isinstance(aspect_ratio, list) and len(aspect_ratio) == 2 and is_number(aspect_ratio[0]) \
        and is_number(aspect_ratio[1]), f"Incorrect format for aspect ratio of module {name}"

    assert 0 <= aspect_ratio[0] <= 1 <= aspect_ratio[
        1], f"Incorrect value for aspect ratio of module {name}"
    return AspectRatio(float(aspect_ratio[0]), float(aspect_ratio[1]))


def parse_yaml_rectangles(rectangles: list, fixed: bool = False, hard: bool = False) -> list[Rectangle]:
    """Parses the rectangles of a module
    :param rectangles: list of rectangles
    :param fixed: are the rectangles fixed
    :param hard: are the rectangles hard
    :return: a list of rectangles (empty if no rectangles are specified)
    """

    rlist = rectangles
    assert isinstance(rlist, list) and len(
        rlist) > 0, f"Incorrect specification of rectangles"
    if is_number(rlist[0]):
        rlist = [rlist]  # List with only one rectangle

    rect_list = []
    for r in rlist:
        rect_list.append(parse_yaml_rectangle(r, fixed, hard))
    return rect_list


def parse_yaml_edges(edges: list[list]) -> list[NamedHyperEdge]:
    """
    Parses the edges of the netlist
    :param edges: a YAML description of the edges (list of edges). Each edge is a list
    :return: the list of edges
    """
    assert isinstance(edges, list), "Incorrect format for the list of edges"
    _edges: list[NamedHyperEdge] = []
    error_str = "Incorrect specification of edge"
    for e in edges:
        assert isinstance(e, list) and len(e) >= 2, error_str
        # We do not need to check for valid identifiers. It will be done elsewhere
        has_weight = is_number(e[-1])
        for i in range(len(e) - 1):
            assert isinstance(e[i], str), error_str
        if not has_weight:
            assert isinstance(e[-1], str), error_str
        weight = 1.0 if not has_weight else float(e[-1])
        modules = e[:-1] if has_weight else e[:]
        _edges.append(NamedHyperEdge(modules, weight))
    return _edges
