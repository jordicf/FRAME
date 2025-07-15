# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""FRAME command-line utility."""

import argparse

import tools.early_router.main_router
import tools.floorset_parser.floorset_handler
import tools.hello.hello  # Fake tool
import tools.draw.draw  # To draw floorplans
import tools.netgen.netgen  # Netlist generator
# To find an initial position of modules using spectral methods
import tools.spectral.spectral
# To improve the initial position of modules using force-directed methods
import tools.force.force
# To find the allocation of the modules using optimization methods
import tools.glbfloor.glbfloor
import tools.rect.rect  # To normalize fuzzy module assignments
import tools.legalfloor.legalfloor  # To legalize the floorplan
import tools.all.all # To execute the whole flow
import tools.uscs_parser.uscs_parser

# Tool names and the entry function they execute.
# The functions must accept two parameters: the tool name and the
# command-line arguments passed to it.
TOOLS = {"hello": tools.hello.hello.main,
         "draw": tools.draw.draw.main,
         "netgen": tools.netgen.netgen.main,
         "spectral": tools.spectral.spectral.main,
         "force": tools.force.force.main,
         "glbfloor": tools.glbfloor.glbfloor.main,
         "rect": tools.rect.rect.main,
         "legalfloor": tools.legalfloor.legalfloor.main,
         "parse_floorset": tools.floorset_parser.floorset_handler.main,
         "uscs_parser": tools.uscs_parser.uscs_parser.main,
         "early_router": tools.early_router.main_router.main,
         "all": tools.all.all.main
         }


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(prog="frame")
    parser.add_argument("tool", choices=TOOLS.keys(),
                        nargs=argparse.REMAINDER, help="tool to execute")
    args = parser.parse_args()

    if args.tool:
        tool_name, tool_args = args.tool[0], args.tool[1:]
        if tool_name in TOOLS:
            TOOLS[tool_name](f"frame {tool_name}", tool_args)
        else:
            print("Unknown tool", tool_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
