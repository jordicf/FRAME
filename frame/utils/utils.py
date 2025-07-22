# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Some common utils to read/write files and handle identifiers and numbers
"""

import pathlib
from enum import Enum
import numbers
import re
from typing import Any, Optional
import yaml
import json

Vector = list[float]
Matrix = list[Vector]
Python_object = object
# Python_object = dict[str, Any] | list[Any]
TextIO_String = str | Python_object


class StrFileType(Enum):
    """File type according to its contents"""

    JSON = 1  # It's a JSON string
    YAML = 2  # It's a YAML string
    FILE = 3  # It's a file (neither JSON nor YAML)
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
    _valid_id = "^[A-Za-z_][A-Za-z0-9_]*"
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


def almost_eq(v1: float, v2: float, epsilon: float = 10e-12) -> bool:
    """Compares two float numbers for equality with a margin of tolerance
    :param v1: one of the numbers
    :param v2: the other number
    :param epsilon: tolerance
    :return: True if they are almost equal, and False otherwise"""
    return abs(v1 - v2) < epsilon


def single_line_string(s: str) -> bool:
    """Checks whether the string has one line only.

    Args:
        s (str): input string

    Returns:
        bool: True if it has only one line, False otherwise
    """
    return s.count("\n") == 0


def string_file_type(s: str) -> StrFileType:
    """It determines the type of contents of the string it can be a file name,
    a JSON string or a YAML string. If none of them is identified,

    Args:
        s (str): input string

    Returns:
        StrFileType: the type of file corresponding to the file
    """

    forbidden = ["\n", "{", "["]
    if all(s.count(c) == 0 for c in forbidden):
        return StrFileType.FILE  # One line without JSON/YAML characters

    try:
        # yaml.safe_load(s, yaml.CLoader)
        yaml.safe_load(s)
        return StrFileType.YAML
    except yaml.YAMLError as e:  # noqa: E722
        try:
            json.loads(s)
            return StrFileType.JSON
        except:  # noqa: E722
            return StrFileType.UNKNOWN


def read_json_yaml_file(filename: str) -> Python_object:
    """
    Reads a JSON or YAML file. It raises an exception in case an error is
    produced incorrect. The type of the file is determined by the suffix of the
    filename (.yaml or .yml for YAML and .json for JSON).
    :param filename: the input file.
    :return: the Python object
    """
    # Check the type of file by suffix
    fname = pathlib.Path(filename)
    str_fname = str(fname)
    suffix = fname.suffix

    if suffix == ". json":
        with open(str_fname, "r") as f:
            return json.load(f)

    if suffix in [".yaml", ".yml"]:
        with open(str_fname, "r") as f:
            return yaml.safe_load(f)

    raise NameError(f"Unknown suffix for file {str_fname}")


def read_json_yaml_text(text: str, is_json: bool = False) -> Python_object:
    """
    Reads a JSON or YAML text. It raises an exception in case an error is
    produced incorrect.
    :param text: the input text
    :param is_json: indicates whether the text is in JSON (True) or YAML (False)
    :return: the Python object
    """
    return json.loads(text) if is_json else yaml.safe_load(text)


def write_json_yaml(
    data: Any, is_json: bool = True, filename: Optional[str] = None
) -> Optional[str]:
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

    with open(filename, "w") as stream:  # dump into a file
        if is_json:
            json.dump(data, stream)
        else:
            yaml.dump(data, stream, default_flow_style=False, indent=4)
        return None
