"""FRAME command-line utility."""

import argparse

import tools.hello.hello
import tools.draw.draw

# Tool names and the entry function they execute.
# The functions must accept two parameters: the tool name and the command-line arguments passed to it.
TOOLS = {"hello": tools.hello.hello.main,
         "draw": tools.draw.draw.main}


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(prog="frame")
    parser.add_argument("tool", choices=TOOLS.keys(), nargs=argparse.REMAINDER, help="tool to execute")
    args = parser.parse_args()

    if args.tool:
        tool_name, tool_args = args.tool[0], args.tool[1:]
        TOOLS[tool_name](f"frame {tool_name}", tool_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
