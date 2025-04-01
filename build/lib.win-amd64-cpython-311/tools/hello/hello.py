# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""Says hello."""
import argparse
from typing import Any


def hello(name: str | None = None) -> str:
    """
    Returns a string saying hello
    :param name: Optional name to say hello to someone in particular
    :return: None
    """
    if name is None:
        return "Hello!"
    return f"Hello {name}!"


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, description="Says hello.")
    parser.add_argument("-n", "--name", type=str, help="name to say hello")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)
    name = options["name"]
    print(hello(name))


if __name__ == "__main__":
    main()
