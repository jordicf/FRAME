from keywords import KW_AREA, KW_FIXED, KW_CENTER, KW_MIN_SHAPE, KW_RECTANGLES, KW_GROUND
from block import Block
from geometry import Rectangle
from yaml_read_netlist import YamlDict, YamlSeq
from netlist_types import HyperEdge


def dump_yaml_blocks(blocks: list[Block]) -> YamlDict:
    """
    Generates a data structure for the blocks that can be dumped in YAML
    :param blocks: list of blocks (see Netlist)
    :return: the data structure
    """
    return {b.name: dump_yaml_block(b) for b in blocks}


def dump_yaml_edges(edges: list[HyperEdge]) -> YamlSeq:
    """
    Generates a data structure for the edges that can be dumped in YAML
    :param edges: list of edges (see Netlist)
    :return: the data structure
    """
    out_edges: YamlSeq = []
    for e in edges:
        edge: list[str | float] = [b.name for b in e.blocks]
        if e.weight != 1:
            edge.append(e.weight)
        out_edges.append(edge)
    return out_edges


def dump_yaml_block(block: Block) -> YamlDict:
    """
    Generates a data structure for the block that can be dumped in YAML
    :param block: a block
    :return: the data structure
    """
    info: dict[str, float | bool | YamlSeq | YamlDict] = {}

    if len(block.area_regions) == 1 and KW_GROUND in block.area_regions:
        info[KW_AREA] = block.area(KW_GROUND)
    else:
        info[KW_AREA] = block.area()

    if block.fixed:
        info[KW_FIXED] = True

    if block.center is not None:
        info[KW_CENTER] = [block.center.x, block.center.y]

    if block.min_shape is not None:
        info[KW_MIN_SHAPE] = [block.min_shape.w, block.min_shape.h]

    if len(block.rectangles) > 0:
        info[KW_RECTANGLES] = dump_yaml_rectangles(block.rectangles)

    return info


def dump_yaml_rectangles(rectangles: list[Rectangle]) -> list[list[float | str]]:
    """
    Generates a list of rectangles to be dumped in YAML
    :param rectangles: list of rectangles
    :return: the list of rectangles
    """
    rects = []
    for r in rectangles:
        list_rect = [r.center.x, r.center.y, r.shape.w, r.shape.h]
        if r.region != KW_GROUND:
            list_rect.append(r.region)
        rects.append(list_rect)
    return rects
