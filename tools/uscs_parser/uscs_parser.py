# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from argparse import ArgumentParser
from tools.uscs_parser.blocks_parser import parse_blocks
from tools.uscs_parser.nets_parser import parse_nets
from tools.uscs_parser.pl_parser import parse_pls
import typing
from frame.utils.utils import write_yaml


def fuse(*objs: dict) -> dict:
    """
    Recursively fuses dictionaries together
    :param objs: The dictionaries
    :return: The fused dictionary
    """
    def couple(d1: dict, d2: dict) -> None:
        for key in d2:
            val = d2[key]
            if isinstance(val, dict):
                if key not in d1:
                    d1[key] = {}
                couple(d1[key], val)
            else:
                d1[key] = val
    ret: dict = {}
    for obj in objs:
        couple(ret, obj)
    return ret


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, typing.Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="USCS dice format parser",
                            usage='%(prog)s [options]')
    parser.add_argument("filename_blocks", type=str,
                        help="Input file 1 (.blocks)")
    parser.add_argument("filename_nets", type=str,
                        help="Input file 2 (.nets)")
    parser.add_argument("filename_pl", type=str,
                        help="Input file 3 (.pl)")
    parser.add_argument("--output", dest="output", default=None, type=str,
                        help="(optional) Output file")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    options = parse_options(prog, args)
    obj1 = parse_blocks(options['filename_blocks'])
    obj2 = parse_nets(options['filename_nets'])
    obj3 = parse_pls(options['filename_pl'])
    result = fuse(obj1, obj2, obj3)
    if options['output'] is None:
        print(write_yaml(result))
    else:
        write_yaml(result, options['output'])
    return 1


if __name__ == "__main__":
    main()
