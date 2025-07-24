# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from typing import Any
from .module import Module
from .netlist_types import HyperEdge, NamedHyperEdge
from ..geometry.geometry import Rectangle
from ..utils.keywords import KW


def dump_yaml_modules(modules: list[Module]) -> dict[str, Any]:
    """
    Generates a data structure for the modules that can be dumped in YAML
    :param modules: list of modules (see Netlist)
    :return: the data structure
    """
    return {b.name: dump_yaml_module(b) for b in modules}


def dump_yaml_edges(edges: list[HyperEdge]) -> list[list[str | float]]:
    """
    Generates a data structure for the edges that can be dumped in YAML
    :param edges: list of edges (see Netlist)
    :return: the data structure
    """
    out_edges = list[list[str | float]]()
    for e in edges:
        edge: list[str | float] = [b.name for b in e.modules]
        if e.weight != 1:
            edge.append(e.weight)
        out_edges.append(edge)
    return out_edges


def dump_yaml_module(module: Module) -> dict:
    """
    Generates a data structure for the module that can be dumped in YAML
    :param module: a module
    :return: the data structure
    """
    info = dict[str, float | bool | list | dict]()

    # Dump all the information
    if not module.is_hard:
        if len(module.area_regions) == 1 and KW.GROUND in module.area_regions:
            info[KW.AREA] = module.area(KW.GROUND)
        else:
            info[KW.AREA] = module.area()

        if module.center is not None:
            info[KW.CENTER] = [module.center.x, module.center.y]

        if module.aspect_ratio is not None:
            info[KW.ASPECT_RATIO] = [
                module.aspect_ratio.min_wh, module.aspect_ratio.max_wh]

    # If it is fixed, no need to say it is hard
    if module.is_fixed:
        info[KW.FIXED] = True
    elif module.is_hard:
        info[KW.HARD] = True

    # In case it is a terminal, no need to say it is hard.
    # The terminal can be written (if it exists)
    if module.is_terminal:
        info[KW.TERMINAL] = True
        info.pop(KW.HARD, None)
        if module.center is not None:
            info[KW.CENTER] = [module.center.x, module.center.y]

    if len(module.rectangles) > 0:
        info[KW.RECTANGLES] = dump_yaml_rectangles(module.rectangles)

    return info


def dump_yaml_rectangles(rectangles: list[Rectangle]) \
        -> list[list[float | str]]:
    """
    Generates a list of rectangles to be dumped in YAML
    :param rectangles: list of rectangles
    :return: the list of rectangles
    """
    rects = list[list[float | str]]()
    for r in rectangles:
        list_rect: list[float | str] = [
            r.center.x, r.center.y, r.shape.w, r.shape.h]
        if r.region != KW.GROUND:
            list_rect.append(r.region)
        rects.append(list_rect)
    return rects


def dump_yaml_namededges(named_edges: list[NamedHyperEdge]) -> list:
    """
    Generates a data structure for the named edges that can be dumped in YAML
    :param named_edges: list of named hyper edges (see Netlist types)
    :return: the data structure
    """
    out_edges = list[list[str | float]]()
    for e in named_edges:
        edge: list[str | float] = [m for m in e.modules]
        if e.weight != 1:
            edge.append(e.weight)
        out_edges.append(edge)
    return out_edges
