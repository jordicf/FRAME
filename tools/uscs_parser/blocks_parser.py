# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from argparse import ArgumentParser

import typing
import re

from frame.utils.utils import write_yaml

Modules = dict[str, dict[str, typing.Any]]
Headers = dict[str, typing.Any]


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, typing.Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description=".blocks parser",
                            usage='%(prog)s [options]')
    parser.add_argument("filename", type=str,
                        help="Allocation input file (.blocks)")
    return vars(parser.parse_args(args))


def blank_line(line: str):
    if len(line) == 0:
        return True
    if line[0] == '#':
        return True
    return False


def word_split(line: str) -> list[str]:
    return re.split(r'[^\S\n\t]+', line)


def parse_terminals(lines: list[str], i: int, modules: Modules):
    # For now, I don't know what to do with this
    # So, TODO, this.
    if blank_line(lines[i]):
        return True
    return False


def parse_rectangles(lines: list[str], i: int, modules: Modules):
    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    # This separation of "soft" + "rectangular" is dumb, but it's the only way to get
    # PyCharm's typo detector to shut up
    if len(words) >= 5 and words[1] == "soft" + "rectangular":
        modules[words[0]] = {
            'area': float(words[2]),
            'min_aspect_ratio': float(words[3]),
            'max_aspect_ratio': float(words[4])
        }
        return True
    elif len(words) == 11 and words[1] == "hard" + "rectilinear":
        # TODO: This.
        return True
    else:
        return False


def parse_header(lines: list[str], i: int, headers: Headers):
    defined_headers = {
        'NumSoftRectangularBlocks': (int, 0),
        'NumHardRectilinearBlocks': (int, 0),
        'NumTerminals': (int, 0)
    }

    if len(headers.keys()) == 0:
        for header in defined_headers.keys():
            headers[header] = defined_headers[header][1]

    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if words[1] != ':':
        return False
    if len(words) != 3:
        raise Exception("Error parsing line " + str(i) + ": Unknown Header:" + lines[i])
    if words[0] in defined_headers:
        headers[words[0]] = defined_headers[words[0]][0](words[2])
    else:
        headers[words[0]] = words[2]
    return True


def parse_blocks(file_path: str):
    modules: Modules = {}
    headers: Headers = {}
    f = open(file_path, "r")
    raw_text = f.read()
    lines = re.split('\n', raw_text)
    i = 1
    while parse_header(lines, i, headers):
        i = i + 1
    while parse_rectangles(lines, i, modules):
        i = i + 1
    while parse_terminals(lines, i, modules):
        i = i + 1
    print(headers)
    return modules


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    options = parse_options(prog, args)
    blocks = parse_blocks(options['filename'])
    print(write_yaml({'Modules': blocks}))
    return 1


if __name__ == "__main__":
    main()
