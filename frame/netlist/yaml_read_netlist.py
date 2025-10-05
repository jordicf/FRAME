# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Module to read netlists in JSON/YAML format
"""

from typing import Any, cast
from frame.netlist.module import Module
from frame.netlist.netlist_types import NamedHyperEdge
from frame.geometry.geometry import (
    Rectangle,
    AspectRatio,
    Point,
    parse_yaml_rectangle,
)
from frame.utils.keywords import KW
from frame.utils.utils import (
    valid_identifier,
    is_number,
    single_line_string,
    read_json_yaml_file,
    read_json_yaml_text,
)


def parse_yaml_netlist(stream: str) -> tuple[list[Module], list[NamedHyperEdge]]:
    """
    Parses a netlist from a file (JSON or YAML) or from a string of text (YAML).
    If the text has only one word, it is assumed to be a file name
    :param stream: name of the file or YAML text
    :return: the list of modules and the list of edges.
    """
    if single_line_string(stream):
        tree = read_json_yaml_file(stream)
    else:
        tree = read_json_yaml_text(stream)

    assert isinstance(tree, dict), "The YAML root node is not a dictionary"
    modules = list[Module]()
    edges = list[NamedHyperEdge]()
    for key, value in tree.items():
        assert key in [KW.MODULES, KW.NETS], f"Unknown key {key}"
        if key == KW.MODULES:
            modules = parse_yaml_modules(value)
        elif key == KW.NETS:
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
    assert isinstance(modules, dict), "The YAML node for modules is not a dictionary"
    _modules = list[Module]()
    for name, module_info in modules.items():
        assert valid_identifier(name), f"Invalid module name: {name}"
        _modules.append(parse_yaml_module(name, module_info))
    return _modules


def parse_yaml_module(name: str, info: dict[str, Any]) -> Module:
    """
    Parses the information of a module
    :param name: Name of the module
    :param info: Information of the module
    :return: a module
    """

    assert valid_identifier(name), f"Invalid name for module: {name}"
    assert isinstance(info, dict), (
        f"The information for module {name} is not a dictionary"
    )
    params = dict[str, Any]()
    for key, value in info.items():
        assert isinstance(key, str)
        if key in [KW.AREA, KW.LENGTH, KW.IO_PIN, KW.TERMINAL, KW.FIXED, KW.HARD, KW.FLIP]:
            params[key] = value
        elif key == KW.CENTER:
            params[KW.CENTER] = parse_yaml_center(value, name)
        elif key == KW.ASPECT_RATIO:
            params[KW.ASPECT_RATIO] = parse_yaml_aspect_ratio(value, name)
        elif key == KW.RECTANGLES:
            pass
        else:
            assert False, f"Unknown module attribute {key}"

    # IO_PIN and TERMINAL are synonyms: let us check that they are not both present
    assert not (KW.IO_PIN in params and KW.TERMINAL in params), (
        f"Module {name}: io_pin and terminal are synonyms, only one of them should be used"
    )
    # We need to anticipate fixed and hard for the rectangles (not a nice code)
    assert KW.FIXED not in params or isinstance(params[KW.FIXED], bool), (
        f"Module {name}: incorrect value for fixed (should be a boolean)"
    )
    assert KW.HARD not in params or isinstance(params[KW.HARD], bool), (
        f"Module {name}: incorrect value for hard (should be a boolean)"
    )
    fixed = KW.FIXED in params and params[KW.FIXED]
    hard = fixed or (KW.HARD in params and params[KW.HARD])
    if KW.RECTANGLES in info:
        params[KW.RECTANGLES] = parse_yaml_rectangles(info[KW.RECTANGLES], fixed, hard)

    return Module(name, **params)


def parse_yaml_center(center: list[float], name: str) -> Point:
    """
    Parses the center of a module
    :param center: module center [x, y]
    :param name: name of the block
    :return: a Point with (x,y)
    """
    assert (
        isinstance(center, list)
        and len(center) == 2
        and is_number(center[0])
        and is_number(center[1])
    ), f"Incorrect format for the center of module {name}"
    return Point(float(center[0]), float(center[1]))


def parse_yaml_aspect_ratio(
    aspect_ratio: float | list[float], name: str
) -> AspectRatio:
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

    assert (
        isinstance(aspect_ratio, list)
        and len(aspect_ratio) == 2
        and is_number(aspect_ratio[0])
        and is_number(aspect_ratio[1])
    ), f"Incorrect format for aspect ratio of module {name}"

    assert 0 <= aspect_ratio[0] <= 1 <= aspect_ratio[1], (
        f"Incorrect value for aspect ratio of module {name}"
    )
    return AspectRatio(float(aspect_ratio[0]), float(aspect_ratio[1]))


def parse_yaml_rectangles(
    rectangles: list, fixed: bool = False, hard: bool = False
) -> list[Rectangle]:
    """Parses the rectangles of a module
    :param rectangles: list of rectangles
    :param fixed: are the rectangles fixed
    :param hard: are the rectangles hard
    :return: a list of rectangles (empty if no rectangles are specified)
    """

    rlist = rectangles
    assert isinstance(rlist, list) and len(rlist) > 0, (
        f"Incorrect specification of rectangles"
    )
    if is_number(rlist[0]):
        rlist = [rlist]  # List with only one rectangle

    rect_list = list[Rectangle]()
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
    _edges = list[NamedHyperEdge]()
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
