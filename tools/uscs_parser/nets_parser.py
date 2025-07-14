# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from argparse import ArgumentParser

import typing
import re

from frame.utils.utils import write_json_yaml

Net = list[typing.Any]
Nets = list[Net]
Headers = dict[str, typing.Any]


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
    parser = ArgumentParser(prog=prog, description=".nets parser",
                            usage='%(prog)s [options]')
    parser.add_argument("filename", type=str,
                        help="Allocation input file (.nets)")
    return vars(parser.parse_args(args))


def parse_header(lines: list[str], i: int, headers: Headers):
    defined_headers = {
        'NumNets': (int, 0),
        'NumPins': (int, 0)
    }

    if len(headers.keys()) == 0:
        for header in defined_headers.keys():
            headers[header] = defined_headers[header][1]

    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if len(words) < 2 or words[1] != ':' or words[0] == 'NetDegree':
        return False
    if len(words) != 3:
        raise Exception("Error parsing line " + str(i+1) +
                        ": Unknown Header:" + lines[i])
    if words[0] in defined_headers:
        headers[words[0]] = defined_headers[words[0]][0](words[2])
    else:
        headers[words[0]] = words[2]
    return True


def parse_net_line(lines: list[str], i: int, net: Net):
    if blank_line(lines[i]):
        return False
    words = word_split(lines[i])
    if len(words) < 1:
        raise Exception("Line number " + str(i+1) + " is empty!")
    net.append(words[0])
    # TODO: Process the rest of arguments
    return True


def parse_net(lines: list[str], i: int, nets: Nets) -> tuple[bool, int]:
    if blank_line(lines[i]):
        return True, i+1
    words = word_split(lines[i])
    if len(words) != 3 or words[0] != 'NetDegree' or words[1] != ":":
        raise Exception("Unknown format on line " + str(i+1) + ": " + lines[i])
    net_size = int(words[2])
    net: Net = []
    i += 1
    j = 0
    while j < net_size:
        if parse_net_line(lines, i, net):
            j += 1
        i += 1
    nets.append(net)
    return True, i


def parse_nets(file_path: str):
    nets: Nets = []
    headers: Headers = {}
    f = open(file_path, "r")
    raw_text = f.read()
    lines = re.split('\n', raw_text)
    i = 1
    while i < len(lines) and parse_header(lines, i, headers):
        i += 1
    while i < len(lines):
        cont, i = parse_net(lines, i, nets)
        if not cont:
            break
    return {'Nets': nets}


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    options = parse_options(prog, args)
    nets = parse_nets(options['filename'])
    print(write_json_yaml(nets, False))
    return 1


if __name__ == "__main__":
    main()
