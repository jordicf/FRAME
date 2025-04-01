# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Some common utils to read/write files and handle identifiers and numbers
"""

import numbers
import re
from typing import Any, TextIO
from ruamel.yaml import YAML
from io import StringIO

Vector = list[float]
Matrix = list[Vector]
YAML_tree = dict[str, Any] | list[Any]
TextIO_String = TextIO | str | YAML_tree


def valid_identifier(ident: Any) -> bool:
    """
    Checks whether the argument is a string and is a valid identifier.
    The first character must be a letter or '_'. The remaining characters can also be digits
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


def read_yaml(stream: TextIO_String) -> YAML_tree:
    """
    Reads a YAML contents from a file or a string. The distinction between a YAML contents and a file name is
    done by checking that ': ' exists in the string. The input can also be a YAML tree
    :param stream: the input. It can be either a file handler, a file name or a YAML contents (in text or tree)
    :return: the YAML tree
    """
    if isinstance(stream, (list, dict)):
        # Nothing to do
        return stream

    if isinstance(stream, str):
        if stream.find(": ") >= 0:
            txt = stream
        else:
            with open(stream) as f:
                txt = f.read()
    else:
        assert isinstance(stream, TextIO)
        txt = stream.read()

    yaml = YAML(typ='safe')
    return yaml.load(txt)


def write_yaml(data: Any, filename: str = None) -> None | str:
    """
    Writes the data into a YAML file. If no file name is given, a string with the yaml contents
    is returned
    :param data: data to be written
    :param filename: name of the output file
    :return: the YAML string in case filename is None
    """

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
        return None


def almost_eq(v1: float, v2: float, epsilon: float = 10e-12) -> bool:
    """Compares two float numbers for equality with a margin of tolerance
    :param v1: one of the numbers
    :param v2: the other number
    :param epsilon: tolerance
    :return: True if they are almost equal, and False otherwise"""
    return abs(v1 - v2) < epsilon
