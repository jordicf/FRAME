"""
Module to read netlists in yaml format
"""

from typing import Any, TextIO, cast

from ruamel.yaml import YAML

from ..utils.keywords import KW_RECTANGLES, KW_CENTER, KW_SHAPE, KW_FIXED, KW_REGION, \
    KW_MODULES, KW_EDGES, KW_AREA, KW_MIN_SHAPE
from ..utils.utils import valid_identifier, is_number
from ..geometry.geometry import Rectangle, Shape, Point

from module import Module
from netlist_types import NamedHyperEdge

YamlDict = dict
YamlSeq = list


def parse_yaml_netlist(stream: str | TextIO, from_text: bool = False) -> tuple[list[Module], list[NamedHyperEdge]]:
    """
    Parses a YAML netlist from a file or from a string of text
    :param stream: name of the YAML file or handle to the file
    :param from_text: if asserted, the netlist is parsed directly from a string of text
    :return: the list of modules and the list of edges.
    """

    if isinstance(stream, str):
        if from_text:
            txt = stream
        else:
            with open(stream) as f:
                txt = f.read()
    else:
        assert isinstance(stream, TextIO)
        txt = stream.read()

    yaml = YAML(typ='safe')
    tree = yaml.load(txt)
    assert isinstance(tree, YamlDict), "The YAML root node is not a dictionary"
    modules: list[Module] = []
    edges: list[NamedHyperEdge] = []
    for key, value in tree.items():
        assert key in [KW_MODULES, KW_EDGES], f"Unknown key {key}"
        if key == KW_MODULES:
            modules = parse_yaml_modules(value)
        elif key == KW_EDGES:
            edges = parse_yaml_edges(value)
        else:
            assert False  # Should never happen
    return modules, edges


def parse_yaml_modules(modules: YamlDict) -> list[Module]:
    """
    Parses the modules of the netlist
    :param modules: The collection of modules
    :return: the list of modules
    """
    assert isinstance(modules, YamlDict), "The YAML node for modules is not a dictionary"
    _modules: list[Module] = []
    for name, module_info in modules.items():
        assert valid_identifier(name), f"Invalid module name: {name}"
        _modules.append(parse_yaml_module(name, module_info))
    return _modules


def parse_yaml_module(name: str, info: YamlDict) -> Module:
    """
    Parses the information of a module
    :param name: Name of the module
    :param info: Information of the module
    :return: a module
    """
    assert isinstance(info, YamlDict), f"The YAML node for module {name} is not a dictionary"
    assert valid_identifier(name), f"Invalid name for module: {name}"

    params: dict[str, Any]
    params = {KW_FIXED: False}  # Parameters for the constructor
    for key, value in info.items():
        assert isinstance(key, str)
        if key == KW_AREA:
            params[key] = value  # The same structure as in YAML
        elif key == KW_CENTER:
            params[KW_CENTER] = parse_yaml_center(value, name)
        elif key == KW_MIN_SHAPE:
            params[KW_MIN_SHAPE] = parse_yaml_min_shape(value, name)
        elif key == KW_FIXED:
            params[KW_FIXED] = parse_yaml_fixed(value, name)
        elif key == KW_RECTANGLES:
            pass
        else:
            assert False, f"Unknown module attribute {key}"

    b = Module(name, **params)

    if KW_RECTANGLES in info:
        rectangles = parse_yaml_rectangles(info[KW_RECTANGLES], cast(bool, params[KW_FIXED]), name)
        for r in rectangles:
            b.add_rectangle(r)
        b.name_rectangles()
    return b


def parse_yaml_center(center: list[float], name: str) -> Point:
    """
    Parses the center of a module
    :param center: module center [x, y]
    :param name: name of the boock
    :return: a Point with (x,y)
    """
    assert isinstance(center, list) and len(center) == 2 and is_number(center[0]) and is_number(center[1]), \
        f"Incorrect format for the center of module {name}"
    return Point(float(center[0]), float(center[1]))


def parse_yaml_min_shape(min_shape: float | list[float], name: str) -> Shape:
    """
    Parses the min size of the module
    :param min_shape: module attributes
    :param name: name of the module
    :return: a Shape with (min width, min height)
    """
    if is_number(min_shape):
        minshape = cast(float, min_shape)
        assert minshape > 0, f"Incorrect min shape for module {name}"
        return Shape(float(minshape), float(minshape))

    assert isinstance(min_shape, list) and len(min_shape) == 2 and is_number(min_shape[0]) \
           and is_number(min_shape[1]), f"Incorrect format for min shape of module {name}"

    assert min_shape[0] >= 0 and min_shape[1] >= 0, f"Incorrect value for min shape of module {name}"
    return Shape(float(min_shape[0]), float(min_shape[1]))


def parse_yaml_fixed(fixed: bool, name: str) -> bool:
    """
    Parses the fixed attribute
    :param fixed: module attribute
    :param name: name of the module
    :return: the value of the attribute fixed (boolean)
    """
    assert isinstance(fixed, bool), f"Incorrect value for the fixed attribute of module {name}"
    return fixed


def parse_yaml_rectangles(rectangles: YamlSeq, fixed: bool, name: str) -> list[Rectangle]:
    """Parses the rectangles of a module
    :param rectangles: sequence of rectangles
    :param fixed: are the rectangles fixed
    :param name: name of the module
    :return: a list of rectangles (empty if no rectangles are specified)
    """

    rlist = rectangles
    assert isinstance(rlist, YamlSeq) and len(rlist) > 0, f"Incorrect specification of rectangles in module {name}"
    if is_number(rlist[0]):
        rlist = [rlist]  # List with only one rectangle

    rect_list = []
    for r in rlist:
        rect_list.append(parse_yaml_rectangle(r, fixed, name))
    return rect_list


def parse_yaml_rectangle(r: YamlSeq, fixed: bool, name: str) -> Rectangle:
    """Parses a rectangle
    :param r: a YAML description of the rectangle (a list with 4 values)
    Optionally, it may contain a fifth parameter (string) specifying a region
    :param fixed: Indicates wheter the rectangle should be fixed
    :param name: name of the module
    :return: a rectangle
    """
    assert isinstance(r, list) and 4 <= len(r) <= 5, f"Incorrect format for rectangle in module {name}"
    for i in range(4):
        assert is_number(r[i]) and r[i] >= 0, f"Incorrect value for rectangle in module {name}"
    if len(r) == 5:
        assert isinstance(r[4], str) and valid_identifier(r[4])

    kwargs = {KW_CENTER: Point(r[0], r[1]), KW_SHAPE: Shape(r[2], r[3]), KW_FIXED: fixed}
    if len(r) == 5:
        kwargs[KW_REGION] = r[4]
    return Rectangle(**kwargs)


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
