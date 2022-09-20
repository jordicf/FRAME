# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from argparse import ArgumentParser

import typing
import re

from frame.utils.utils import write_yaml

Modules = dict[str, dict[str, typing.Any]]


def blank_line(line: str):
    if len(line) == 0:
        return True
    if line[0] == '#':
        return True
    return False


def word_split(line: str) -> list[str]:
    return re.split(r'[\s\n\t]+', line)


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, typing.Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description=".pl parser",
                            usage='%(prog)s [options]')
    parser.add_argument("filename", type=str,
                        help="Allocation input file (.pl)")
    return vars(parser.parse_args(args))


def parse_pl(lines: list[str], i: int, modules: Modules) -> bool:
    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if len(words) != 3:
        raise Exception("Don't know how to parse line (" + str(i + 1) + "): " + lines[i])
    modules[words[0]] = {
        'fixed': True,
        'terminal': True,
        'center': [float(words[1]), float(words[2])]
    }
    return True


def parse_pls(file_path: str):
    modules: Modules = {}
    f = open(file_path, "r")
    raw_text = f.read()
    lines = re.split('\n', raw_text)
    i = 1
    while i < len(lines) and parse_pl(lines, i, modules):
        i = i + 1
    return {'Modules': modules}


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    options = parse_options(prog, args)
    modules = parse_pls(options['filename'])
    print(write_yaml(modules))
    return 1


if __name__ == "__main__":
    main()
