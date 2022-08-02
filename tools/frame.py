"""FRAME command-line utility."""

import argparse

import tools.hello.hello  # Fake tool
import tools.draw.draw  # To draw floorplans
import tools.netgen.netgen  # Netlist generator
import tools.spectral.spectral  # To find an initial position of modules using spectral methods
import tools.glbfloor.glbfloor  # ...

# Tool names and the entry function they execute.
# The functions must accept two parameters: the tool name and the command-line arguments passed to it.
TOOLS = {"hello": tools.hello.hello.main,
         "draw": tools.draw.draw.main,
         "netgen": tools.netgen.netgen.main,
         "spectral": tools.spectral.spectral.main,
         "glbfloor": tools.glbfloor.glbfloor.main
         }


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(prog="frame")
    parser.add_argument("tool", choices=TOOLS.keys(), nargs=argparse.REMAINDER, help="tool to execute")
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
