# (c) FRAME Project contributors 2025
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).
"""
Parser that converts `.blocks`, `.nets`, `.pl` triples into FRAME netlist YAML.
"""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
import typing

from frame.utils.utils import write_json_yaml
from tools.uscs_parser.blocks_parser import parse_blocks
from tools.uscs_parser.nets_parser import parse_nets
from tools.uscs_parser.pl_parser import parse_pls

ModulesDict = dict[str, dict[str, typing.Any]]
NetlistDict = dict[str, typing.Any]
DEFAULT_CENTER = [0.0, 0.0]


def parse_options(
    prog: str | None = None,
    args: list[str] | None = None,
) -> dict[str, typing.Any]:
    parser = ArgumentParser(
        prog=prog,
        description="Parse USCS benchmark files into FRAME netlist YAML",
        usage="%(prog)s [options]",
    )
    parser.add_argument("filename_blocks", type=str, help="Input file 1 (.blocks)")
    parser.add_argument("filename_nets", type=str, help="Input file 2 (.nets)")
    parser.add_argument("filename_pl", type=str, help="Input file 3 (.pl)")
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        type=str,
        help="(optional) Output YAML file",
    )
    return vars(parser.parse_args(args))


def _copy_center(center: list[typing.Any] | None) -> list[float]:
    if center is None:
        return deepcopy(DEFAULT_CENTER)
    return [float(center[0]), float(center[1])]


def build_terminal_module(name: str, pl_modules: ModulesDict) -> dict[str, typing.Any]:
    if name not in pl_modules:
        raise ValueError(f"Terminal {name} does not have coordinates in .pl file")
    pl_info = pl_modules[name]
    module = {
        "io_pin": True,
        "length": 0.0,
        "center": _copy_center(pl_info.get("center")),
    }
    # if "fixed" in pl_info:
    #     module["fixed"] = bool(pl_info["fixed"])
    return module


def build_block_module(
    name: str,
    block_info: dict[str, typing.Any],
    pl_modules: ModulesDict,
) -> dict[str, typing.Any]:
    module: dict[str, typing.Any] = {
        "area": float(block_info["area"]),
        "center": _copy_center(pl_modules.get(name, {}).get("center")),
    }
    if "aspect_ratio" in block_info:
        module["aspect_ratio"] = block_info["aspect_ratio"]
    if block_info.get("fixed") is not None:
        module["fixed"] = bool(block_info["fixed"])
    if block_info.get("hard") is not None:
        module["hard"] = bool(block_info["hard"])
    if "rectangles" in block_info:
        module["rectangles"] = deepcopy(block_info["rectangles"])
    return module


def build_modules(
    blocks_modules: ModulesDict,
    pl_modules: ModulesDict,
) -> ModulesDict:
    modules: ModulesDict = {}
    for name, block_info in blocks_modules.items():
        if block_info.get("terminal"):
            modules[name] = build_terminal_module(name, pl_modules)
        else:
            modules[name] = build_block_module(name, block_info, pl_modules)
    for name in pl_modules:
        if name not in modules:
            modules[name] = build_terminal_module(name, pl_modules)
    return modules


def build_netlist(
    blocks: NetlistDict,
    nets: NetlistDict,
    pls: NetlistDict,
) -> NetlistDict:
    modules = build_modules(blocks["Modules"], pls["Modules"])
    return {
        "Modules": modules,
        "Nets": nets["Nets"],
    }


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    options = parse_options(prog, args)
    blocks = parse_blocks(options["filename_blocks"])
    nets = parse_nets(options["filename_nets"])
    pls = parse_pls(options["filename_pl"])
    result = build_netlist(blocks, nets, pls)
    if options["output"] is None:
        print(write_json_yaml(result, False))
    else:
        write_json_yaml(result, False, options["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

