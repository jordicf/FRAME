# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from frame.geometry.geometry import Point, Shape, Rectangle
from frame.utils.keywords import KW
from frame.utils.utils import (
    is_number,
    string_is_number,
    valid_identifier,
    single_line_string,
    read_json_yaml_file,
    read_json_yaml_text
)


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


def parse_yaml_die(stream: str) -> tuple[Rectangle, list[Rectangle]]:
    """
    Parses a die from a file or from a string of text. The string of text
    can have the form <width>x<height> (e.g., 5.5x10) of be a YAML contents.
    :param stream: name of YAML file or string of text
    :return: the bounding box of the die and the list of non-ground rectangles
    """

    # First check for a string of the form <width>x<height>, e.g., 5.5x10
    if isinstance(stream, str):
        shape = string_die(stream)
        if shape is not None:
            die = Rectangle(
                **{KW.CENTER: Point(shape.w / 2, shape.h / 2), KW.SHAPE: shape}
            )
            return die, list[Rectangle]()

    # Check whether the string is a filename (one line) or a YAML text
    if single_line_string(stream):
        tree = read_json_yaml_file(stream)
    else:
        tree = read_json_yaml_text(stream)

    assert isinstance(tree, dict), "The die is not a dictionary"

    for key in tree:
        assert key in [KW.WIDTH, KW.HEIGHT, KW.REGIONS], (
            f"Unknown keyword in die: {key}"
        )

    assert KW.WIDTH in tree and KW.HEIGHT in tree, (
        "Wrong format of the die: Missing width or height"
    )
    shape = Shape(tree[KW.WIDTH], tree[KW.HEIGHT])
    assert is_number(shape.w) and shape.w > 0, "Wrong specification of the die width"
    assert is_number(shape.h) and shape.h > 0, "Wrong specification of the die height"
    die = Rectangle(**{KW.CENTER: Point(shape.w / 2, shape.h / 2), KW.SHAPE: shape})

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
    return die, regions


def parse_die_rectangle(r: list):
    """
    Parses a rectangle
    :param r: a YAML description of the rectangle (a list with 5 values).
    :return: a Rectangle
    """
    assert isinstance(r, list) and len(r) == 5, "Incorrect format of die rectangle"
    for i in range(4):
        assert is_number(r[i]) and r[i] >= 0, "Incorrect value for die rectangle"
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
