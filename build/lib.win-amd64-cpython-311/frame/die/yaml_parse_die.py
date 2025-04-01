# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from frame.geometry.geometry import Point, Shape, Rectangle
from frame.utils.keywords import KW_WIDTH, KW_HEIGHT, KW_REGION, KW_REGIONS, KW_GROUND,\
    KW_BLOCKAGE, KW_CENTER, KW_SHAPE
from frame.utils.utils import is_number, string_is_number, valid_identifier, read_yaml, TextIO_String


def string_die(die: str) -> Shape | None:
    """
    Parses the die string of the form <width>x<height> or from a filename
    :param die: the die string or filename
    :return: a shape if it has the form <width>x<height>, or None otherwise
    """
    numbers = die.rsplit('x')
    if len(numbers) == 2 and string_is_number(numbers[0]) and string_is_number(numbers[1]):
        w, h = float(numbers[0]), float(numbers[1])
        assert w > 0 and h > 0, "The width and height of the layout must be positive"
        return Shape(w, h)
    return None


def parse_yaml_die(stream: TextIO_String) -> tuple[Rectangle, list[Rectangle]]:
    """
    Parses a YAML die from a file or from a string of text
    :param stream: name of the YAML file or handle to the file
    :return: the bounding box of the die and the list of non-ground rectangles
    """

    # First check for a string of the form <width>x<height>, e.g., 5.5x10
    if isinstance(stream, str):
        shape = string_die(stream)
        if shape is not None:
            die = Rectangle(**{KW_CENTER: Point(shape.w/2, shape.h/2), KW_SHAPE: shape})
            return die, []

    # Read a YAML contents
    tree = read_yaml(stream)
    assert isinstance(tree, dict), "The die is not a dictionary"

    for key in tree:
        assert key in [KW_WIDTH, KW_HEIGHT, KW_REGIONS], f"Unknown keyword in die: {key}"

    assert KW_WIDTH in tree and KW_HEIGHT in tree, "Wrong format of the die: Missing width or height"
    shape = Shape(tree[KW_WIDTH], tree[KW_HEIGHT])
    assert is_number(shape.w) and shape.w > 0, "Wrong specification of the die width"
    assert is_number(shape.h) and shape.h > 0, "Wrong specification of the die height"
    die = Rectangle(**{KW_CENTER: Point(shape.w/2, shape.h/2), KW_SHAPE: shape})

    regions: list[Rectangle] = []
    if KW_REGIONS in tree:
        rlist = tree[KW_REGIONS]
        assert isinstance(rlist, list) and len(rlist) > 0, f"Incorrect specification of die rectangles"
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
    assert isinstance(r[4], str) and (valid_identifier(r[4]) or r[4] == KW_GROUND or r[4] == KW_BLOCKAGE), \
        f"Invalid identifier for die region: {r[4]}"
    assert r[4] != KW_GROUND, "Only non-ground regions can be specified in the die"
    kwargs = {KW_CENTER: Point(r[0], r[1]), KW_SHAPE: Shape(r[2], r[3]), KW_REGION: r[4]}
    return Rectangle(**kwargs)
