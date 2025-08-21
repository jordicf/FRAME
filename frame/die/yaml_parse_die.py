# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from typing import Optional
from frame.geometry.geometry import Point, Shape, Rectangle
from frame.utils.keywords import KW
from frame.utils.utils import (
    is_number,
    string_is_number,
    valid_identifier,
    single_line_string,
    read_json_yaml_file,
    read_json_yaml_text,
)

# Type for IO segments (candidates for location of pin arrays)
# It is a dictionary with the name of the IO pin as key
IOsegments = dict[str, list[Rectangle]]


def string_die(die: str) -> Shape | None:
    """
    Parses the die string of the form <width>x<height> or from a filename
    :param die: the die string or filename
    :return: a shape if it has the form <width>x<height>, or None otherwise
    """
    numbers = die.rsplit("x")
    if (
        len(numbers) == 2
        and string_is_number(numbers[0])
        and string_is_number(numbers[1])
    ):
        w, h = float(numbers[0]), float(numbers[1])
        assert w > 0 and h > 0, "The width and height of the layout must be positive"
        return Shape(w, h)
    return None


def parse_yaml_die(
    stream: str,
) -> tuple[Rectangle, list[Rectangle], Optional[IOsegments]]:
    """
    Parses a die from a file or from a string of text. The string of text
    can have the form <width>x<height> (e.g., 5.5x10) of be a YAML contents.
    :param stream: name of YAML file or string of text
    :return: the bounding box of the die, the list of non-ground rectangles and
             the set of IOsegments
    """

    # First check for a string of the form <width>x<height>, e.g., 5.5x10
    if isinstance(stream, str):
        shape = string_die(stream)
        if shape is not None:
            die = Rectangle(
                **{KW.CENTER: Point(shape.w / 2, shape.h / 2), KW.SHAPE: shape}
            )
            return die, list[Rectangle](), None

    # Check whether the string is a filename (one line) or a YAML text
    if single_line_string(stream):
        tree = read_json_yaml_file(stream)
    else:
        tree = read_json_yaml_text(stream)

    assert isinstance(tree, dict), "The die is not a dictionary"

    for key in tree:
        assert key in [KW.WIDTH, KW.HEIGHT, KW.REGIONS, KW.IO_SEGMENTS], (
            f"Die: Unknown keyword {key}"
        )

    assert KW.WIDTH in tree and KW.HEIGHT in tree, (
        "Die wrong format: Missing width or height"
    )
    shape = Shape(tree[KW.WIDTH], tree[KW.HEIGHT])
    assert is_number(shape.w) and shape.w > 0, "Die: wrong specification of the width"
    assert is_number(shape.h) and shape.h > 0, "Die: wrong specification of the height"
    die = Rectangle(**{KW.CENTER: Point(shape.w / 2, shape.h / 2), KW.SHAPE: shape})

    # Get the specialized regions and blockages of the die
    regions = list[Rectangle]()
    if KW.REGIONS in tree:
        rlist = tree[KW.REGIONS]
        assert isinstance(rlist, list) and len(rlist) > 0, (
            f"Incorrect specification of die rectangles"
        )
        if is_number(rlist[0]):
            rlist = [rlist]  # List with only one rectangle
        for r in rlist:
            regions.append(parse_die_rectangle(r))

    # Get the IO segments for the IO pins
    io_segments: Optional[IOsegments] = None
    if KW.IO_SEGMENTS in tree:
        io_segments = IOsegments()
        for name, segments in tree[KW.IO_SEGMENTS].items():
            assert valid_identifier(name), f"Invalid identifier for IO segment: {name}"
            assert isinstance(segments, list) and len(segments) > 0, (
                f"Die: incorrect specification of IO segments for {name}"
            )
            if is_number(segments[0]):
                segments = [segments]  # List with only one rectangle
            assert name not in io_segments, f"Die: Duplicate IO pin: {name}"
            io_segments[name] = list[Rectangle]()
            for s in segments:
                assert isinstance(s, list) and len(s) == 4, (
                    f"Die: incorrect format of IO segment {name}: {s}"
                )
                io_segments[name].append(parse_pin_segment(name, s))

    return die, regions, io_segments


def parse_die_rectangle(r: list) -> Rectangle:
    """
    Parses a rectangle
    :param mod_name: name of the module to which the rectangle belongs
    :param r: a YAML description of the rectangle (a list with 5 values).
    :return: a Rectangle
    """
    assert isinstance(r, list) and len(r) == 5, (
        f"Incorrect format of die rectangle {r}"
    )
    for i in range(4):
        assert is_number(r[i]) and r[i] >= 0, "Incorrect value of die rectangle {r}"
    assert isinstance(r[4], str) and (
        valid_identifier(r[4]) or r[4] == KW.GROUND or r[4] == KW.BLOCKAGE
    ), f"Invalid identifier for die region: {r[4]}"
    assert r[4] != KW.GROUND, "Only non-ground regions can be specified in the die"
    kwargs = {
        KW.CENTER: Point(r[0], r[1]),
        KW.SHAPE: Shape(r[2], r[3]),
        KW.REGION: r[4],
    }
    return Rectangle(**kwargs)


def parse_pin_segment(name: str, r: list) -> Rectangle:
    """
    Parses a segmentle for an IO pin
    :param name: name of the IO pin
    :param r: a YAML description of the segment (a list with 4 values).
    :return: a Rectangle
    """
    assert isinstance(r, list) and len(r) == 4, "Die: incorrect format of pin segment for {name}"
    assert all(is_number(x) and x >= 0 for x in r), "Die: incorrect value of pin segment for {name}"
    kwargs = {
        KW.CENTER: Point(r[0], r[1]),
        KW.SHAPE: Shape(r[2], r[3]),
    }
    rect = Rectangle(**kwargs)
    assert rect.is_line, f"Die: IO segment for {name} is not a segment: {r}"
    return rect
