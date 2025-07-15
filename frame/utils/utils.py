# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Some common utils to read/write files and handle identifiers and numbers
"""

from enum import Enum
import numbers
import re
from typing import Any, TextIO, Optional
import yaml
import json

Vector = list[float]
Matrix = list[Vector]
Python_object = object
# Python_object = dict[str, Any] | list[Any]
TextIO_String = TextIO | str | Python_object


class StrFileType(Enum):
    """File type according to its contents"""
    JSON = 1    # It's a JSON string
    YAML = 2    # It's a YAML string
    FILE = 3    # It's a file (neither JSON nor YAML)
    UNKNOWN = 4  # Unknown type


def valid_identifier(ident: Any) -> bool:
    """
    Checks whether the argument is a string and is a valid identifier.
    The first character must be a letter or '_'.
    The remaining characters can also be digits
    :param ident: identifier.
    :return: True if valid, and False otherwise.
    """
    if not isinstance(ident, str):
        return False
    _valid_id = '^[A-Za-z_][A-Za-z0-9_]*'
    return re.fullmatch(_valid_id, ident) is not None


def is_number(n: Any) -> bool:
    """
    Checks whether a value is a number (int or float).
    :param n: the number.
    :return: True if it is a number, False otherwise.
    """
    return isinstance(n, numbers.Real)


def string_is_number(s: str) -> bool:
    """
    Checks whether a string represents a number.
    :param s: the string.
    :return: True if it represents a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def string_file_type(s: str) -> StrFileType:
    """It determines the type of contents of the string it can be a file name,
    a JSON string or a YAML string. If none of them is identified,

    Args:
        s (str): input string

    Returns:
        StrFileType: the type of file corresponding to the file
    """

    forbidden = ['\n', '{', '[', ':']

    if all(s.count(c) == 0 for c in forbidden):
        return StrFileType.FILE  # One line without JSON/YAML characters

    try:
        yaml.load(s, yaml.CLoader)
        return StrFileType.YAML
    except:  # noqa: E722
        try:
            json.loads(s)
            return StrFileType.JSON
        except:  # noqa: E722
            return StrFileType.UNKNOWN


def read_json_yaml(stream: TextIO_String) -> Python_object:
    """
    Reads JSON or YAML contents from a file or a string. The input can also be
    a JSON/YAML tree. It raises a syntax error in case the string cannot be
    identified with a JSON or YAML file/string.
    :param stream: the input. It can be either a file handler, a file name
                   or a JSON/YAML contents (in text or tree)
    :return: the JSON/YAML tree
    """
    if isinstance(stream, (list, dict)):
        # Nothing to do
        return stream

    if isinstance(stream, str):
        txt = stream
        ftype = string_file_type(stream)
        assert ftype != StrFileType.UNKNOWN, "Unknown JSON/YAML contents"

        if ftype == StrFileType.FILE:  # It's a file name
            with open(stream) as f:
                txt = f.read()
                ftype = string_file_type(txt)

    else:  # It's a TextIO
        assert isinstance(stream, TextIO)
        txt = stream.read()
        ftype = string_file_type(txt)

    assert ftype != StrFileType.FILE
    assert ftype != StrFileType.UNKNOWN, "Unknown JSON/YAML contents"

    # Now we have JSON or YAML text

    if ftype == StrFileType.JSON:
        return json.loads(txt)

    return yaml.safe_load(txt)


def write_json_yaml(data: Any, is_json: bool = True,
                    filename: Optional[str] = None) -> None | str:
    """
    Writes the data into a JSON or YAML file. If no file name is given,
    a string with the yaml contents is returned
    :param data: data to be written
    :param is_json: True if a JSON file is to be generated, otherwise YAML
    :param filename: name of the output file
    :return: the JSON/YAML string in case filename is None
    """

    if filename is None:  # generate an output string
        dump_func = json.dumps if is_json else yaml.dump
        return dump_func(data)

    with open(filename, 'w') as stream:  # dump into a file
        dump_func = json.dump if is_json else yaml.dump
        if is_json:
            json.dump(data, stream)
        else:
            yaml.dump(data, stream, default_flow_style=False, indent=4)
        return None


def almost_eq(v1: float, v2: float, epsilon: float = 10e-12) -> bool:
    """Compares two float numbers for equality with a margin of tolerance
    :param v1: one of the numbers
    :param v2: the other number
    :param epsilon: tolerance
    :return: True if they are almost equal, and False otherwise"""
    return abs(v1 - v2) < epsilon
