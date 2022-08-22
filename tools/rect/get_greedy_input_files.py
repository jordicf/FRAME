import typing
from argparse import ArgumentParser
from tools.rect.rect_io import get_ifile, getfile, selectbox


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, typing.Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="An example generation tool", usage='%(prog)s [options]')
    parser.add_argument("filename", type=str, help="Input file (.yaml)")
    parser.add_argument("outdir",   type=str, help="The output file directory")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """
    Main function.
    """
    options = parse_options(prog, args)
    file: str = options['filename']
    if file[-5:] != ".yaml":
        raise Exception("Input file name should end in .yaml")
    basefile = file.split(sep="/")[-1][:-5]
    ifile = get_ifile(file)

    module_names = set()
    for r in ifile['Rectangles']:
        for bname in r:
            for lst in r[bname][1]['mod']:
                for m in lst:
                    module_names.add(m)

    counter = 10
    for mod in sorted(module_names, key=lambda x: int(x[1:])):
        input_problem, selbox = selectbox(mod, ifile)
        iofile = open(options["outdir"] + basefile + "." + mod + ".txt", "w")
        iofile.write(getfile(input_problem, ifile, 2.00))
        iofile.close()
        counter -= 1
        if counter <= 0:
            break

    return 1


if __name__ == "__main__":
    main()
