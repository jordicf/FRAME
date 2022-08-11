"""
Package to normalize a fuzzy configuration of
modules in a non-uniform rectangular grid into
adjacent rectangles.
"""

import subprocess
from argparse import ArgumentParser

import pseudobool
import satmanager
import typing
from tools.rect.rect_io import get_ifile, getfile, selectbox

from frame.utils.utils import write_yaml
from canvas import Canvas, colmix

# Custom types
SimpleBox = tuple[float, float, float, float]
InputBox = tuple[float, float, float, float, float]
InputProblem = list[InputBox]


class Carrier:
    def __init__(self):
        self.input_problem: InputProblem = []
        self.selbox: str = ""
        self.factor: float = 0.0
        self.inibox: InputBox = (0, 0, 0, 0, 0)
        self.blocks: list[int] = []
        self.prev_x: dict[float, float] = {}
        self.prev_y: dict[float, float] = {}
        self.next_x: dict[float, float] = {}
        self.next_y: dict[float, float] = {}
        self.xcoords: list[float] = []
        self.ycoords: list[float] = []
        self.theoreticalBestArea: float = 0.0


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, typing.Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="An example generation tool", usage='%(prog)s [options]')
    parser.add_argument("filename", type=str,                 help="Input file (.yaml)")
    parser.add_argument("--minarea", dest='f', const=0.89, default=2.00, action='store_const',
                        help="Minimizes the total area while guaranteeing a minimum coverage of the original")
    parser.add_argument("--maxdiff", dest='f', const=3.00, default=2.00, action='store_const',
                        help="Maximizes the difference between the inner area and the outer area")
    parser.add_argument("--minerr",  dest='f', const=2.00, default=2.00, action='store_const',
                        help="Minimizes the error (Default)")
    parser.add_argument("--sf", type=float, dest='f', default=2.00,
                        help="Manually set the factor to number F (Not recommended)")
    parser.add_argument("--plot", dest="plot", const=True, default=False, action="store_const",
                        help="Plots the problem together with the solutions found")
    parser.add_argument("--outfile", type=str, dest='file', default=None,
                        help="The output file path (yaml)")
    parser.add_argument("--module", type=str, dest='module', default=None,
                        help="The module to optimize. If none is introduced, it optimizes all of them")

    return vars(parser.parse_args(args))


def enforce_bb(carrier: Carrier, ifile, sm: satmanager.SATManager, btag: str, cbtag: str):
    """
    Enforces that a certain rectangle to actually be a rectangle.
    It also enforces the rectangle to be adjacent to the trunk rectangle,
    (if the rectangle is not the trunk rectangle already)
    """
    var_lilx: dict[float, pseudobool.Literal] = {}
    var_bigx: dict[float, pseudobool.Literal] = {}
    var_lily: dict[float, pseudobool.Literal] = {}
    var_bigy: dict[float, pseudobool.Literal] = {}
    var_b: dict[int, pseudobool.Literal] = {}
    xcoords: list[float] = carrier.xcoords
    ycoords: list[float] = carrier.ycoords
    blocks: list[int] = carrier.blocks
    input_problem: InputProblem = carrier.input_problem
    prev_x: dict[float, float] = carrier.prev_x
    prev_y: dict[float, float] = carrier.prev_y
    next_x: dict[float, float] = carrier.next_x
    next_y: dict[float, float] = carrier.next_y
    for x in xcoords:
        var_lilx[x] = sm.newvar(btag + "x_" + str(x), "")
        var_bigx[x] = sm.newvar(btag + "X_" + str(x), "")
    for y in ycoords:
        var_lily[y] = sm.newvar(btag + "y_" + str(y), "")
        var_bigy[y] = sm.newvar(btag + "Y_" + str(y), "")
    for b in blocks:
        var_b[b] = sm.newvar(btag + str(b), "")
        sm.imply([var_b[b]], var_lilx[input_problem[b][2]])
        sm.imply([var_b[b]], var_bigx[input_problem[b][0]])
        sm.imply([var_b[b]], var_lily[input_problem[b][3]])
        sm.imply([var_b[b]], var_bigy[input_problem[b][1]])
    for i in range(1, len(xcoords)):
        sm.imply([var_bigx[prev_x[xcoords[i]]]], var_bigx[xcoords[i]])
        sm.imply([var_lilx[xcoords[i]]], var_lilx[prev_x[xcoords[i]]])
    for i in range(1, len(ycoords)):
        sm.imply([var_bigy[prev_y[ycoords[i]]]], var_bigy[ycoords[i]])
        sm.imply([var_lily[ycoords[i]]], var_lily[prev_y[ycoords[i]]])
    for b in blocks:
        sm.imply([var_lilx[next_x[input_problem[b][0]]],
                  var_bigx[prev_x[input_problem[b][2]]],
                  var_lily[next_y[input_problem[b][1]]],
                  var_bigy[prev_y[input_problem[b][3]]]], var_b[b])

    # Enforce connection to the trunk rectangle
    if btag != cbtag:
        north = sm.newvar(btag + "north", "")
        south = sm.newvar(btag + "south", "")
        east = sm.newvar(btag + "east", "")
        west = sm.newvar(btag + "west", "")
        sm.heuleencoding([north, south, east, west])
        sm.pseudoboolencoding(north + south + east + west >= 1)
        for b1 in blocks:
            if input_problem[b1][0] == 0:
                sm.imply([west], -var_b[b1])
            if input_problem[b1][1] == 0:
                sm.imply([north], -var_b[b1])
            if input_problem[b1][2] == int(ifile['Width']):
                sm.imply([east], -var_b[b1])
            if input_problem[b1][3] == int(ifile['Height']):
                sm.imply([south], -var_b[b1])
            for b2 in blocks:
                bb1: tuple[float, float, float, float, float] = input_problem[b1]
                bb2: tuple[float, float, float, float, float] = input_problem[b2]
                if bb1[0] == bb2[2] and bb1[3] > bb2[1] and bb1[1] < bb2[3]:
                    sm.imply([var_b[b1], west, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))
                if bb1[2] == bb2[0] and bb1[3] > bb2[1] and bb1[1] < bb2[3]:
                    sm.imply([var_b[b1], east, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))
                if bb1[1] == bb2[3] and bb1[2] > bb2[0] and bb1[0] < bb2[2]:
                    sm.imply([var_b[b1], north, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))
                if bb1[3] == bb2[1] and bb1[2] > bb2[0] and bb1[0] < bb2[2]:
                    sm.imply([var_b[b1], south, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))


def solve(carrier: Carrier,
          ifile,
          ratio: float,
          dif: tuple[int, int],
          nboxes: int) -> tuple[tuple[int, int], list[SimpleBox]]:
    """
    Generates the SAT problem, calls the solver and returns the solution
    :param carrier: The blocks, input_problem, factor and theoretical_best_area carrier
    :param ifile: The input file (parsed)
    :param ratio: The f hyperparameter
    :param dif: The previous optimal solution (as a rational), for optimization purposes
    :param nboxes: The allowed number of boxes of the solution
    :return: The cost for an existing solution better than the previous one, and the list of rectangles
    """
    sm = satmanager.SATManager()

    blocks: list[int] = carrier.blocks
    input_problem: InputProblem = carrier.input_problem
    factor: float = carrier.factor
    theoretical_best_area: float = carrier.theoreticalBestArea
    selarea = pseudobool.Expr()
    allblocks = pseudobool.Expr()
    vars_b = []
    for b in blocks:
        selarea = selarea + sm.newvar("b_" + str(b), "") * int(area(carrier, b, True))
        allblocks = allblocks + sm.newvar("b_" + str(b), "")
        vars_b.append(- sm.newvar("b_" + str(b), ""))
        lst = []
        for i in range(0, nboxes):
            sm.imply([sm.newvar("b" + str(i) + "_" + str(b), "")], sm.newvar("b_" + str(b), ""))
            lst.append(- sm.newvar("b" + str(i) + "_" + str(b), ""))
        sm.imply(lst, - sm.newvar("b_" + str(b), ""))
    vars_b.sort(key=lambda x: x.v)
    realarea = pseudobool.Expr()
    for b in blocks:
        realarea = realarea + sm.newvar("b_" + str(b), "") * int(area(carrier, b, False))
    obj = ratio * selarea - realarea

    # Have at least one block, please
    sm.pseudoboolencoding(allblocks >= 1)

    # Min area approach
    if ratio < 1:
        sm.pseudoboolencoding(realarea >= round(theoretical_best_area * ratio))
        sm.pseudoboolencoding(selarea * dif[1] - realarea * dif[0] >= 0)
    # Min error approach
    else:
        sm.pseudoboolencoding(obj >= dif[0])

    for i in range(0, nboxes):
        enforce_bb(carrier, ifile, sm, "b" + str(i) + "_", "b" + str(0) + "_")
    if not sm.solve():
        print("Insat")
        return (0, 1), []
    print("Selected area:    " + str(float(sm.evalexpr(selarea)) / float(factor)))
    print("Real area:        " + str(float(sm.evalexpr(realarea)) / float(factor)))
    print("Theoretical area: " + str(theoretical_best_area / float(factor)))
    print("Error objective:  " + str(sm.evalexpr(obj)))

    rects = [(float('inf'), float('inf'), -float('inf'), -float('inf'))] * nboxes
    for i in range(0, nboxes):
        for b in blocks:
            if sm.value(sm.newvar("b" + str(i) + "_" + str(b), "")) == 1:
                (cx0, cy0, cx1, cy1) = rects[i]
                (nx0, ny0, nx1, ny1, p) = input_problem[b]
                if cx0 > nx0:
                    cx0 = nx0
                if cy0 > ny0:
                    cy0 = ny0
                if cx1 < nx1:
                    cx1 = nx1
                if cy1 < ny1:
                    cy1 = ny1
                rects[i] = (cx0, cy0, cx1, cy1)

    # Min area approach
    if ratio < 1:
        return (int(sm.evalexpr(selarea) + 1), int(sm.evalexpr(realarea))), rects
    # Min error approach
    else:
        return (int(sm.evalexpr(obj) + 1), 1), rects


def area(carrier: Carrier, b: int | InputBox, sel: bool) -> float:
    """
    Returns the area of box b
    :param carrier: The input_problem and factor variable carrier
    :param b: The given box, either as an index or as a tuple
    :param sel: Whether we want the total area or just the proportion occupied by the module
    :return: The area
    """
    if isinstance(b, tuple):
        (x1, y1, x2, y2, p) = b
    elif isinstance(b, int):
        (x1, y1, x2, y2, p) = carrier.input_problem[b]
    else:
        raise Exception("Invalid type")
    if sel:
        return carrier.factor * p * (x2 - x1) * (y2 - y1)
    return carrier.factor * (x2 - x1) * (y2 - y1)


def fstr_to_tuple(carrier: Carrier, p: str, f: float) -> tuple[int, int]:
    """
    Turns p, which is a string codifying a float, into a rational
    :param carrier: The inibox variable carrier
    :param p: The input string
    :param f: The f hyperparameter
    :return: The value of p as a rational
    """
    # Min area approach
    if f < 1:
        fp = False
        n1 = 0
        n2 = 1
        for i in range(0, len(p)):
            if p[i] == '.':
                fp = True
            else:
                n1 = n1 * 10 + int(p[i])
                if fp:
                    n2 = n2 * 10
        return n1, n2
    # Min error approach
    print(area(carrier, carrier.inibox, True), area(carrier, carrier.inibox, False))
    return round(f * area(carrier, carrier.inibox, True) - area(carrier, carrier.inibox, False)), 1


def definecoords(carrier: Carrier) -> None:
    """
    Generates a set of auxiliary variables, useful for many parts of the code.
    These auxiliary variables are:
    blocks : A list of the identifiers for the blocks
    xcoords: An ordered list of all x coordinates
    ycoords: An ordered list of all y coordinates
    prev_x : A map from x coordinates to their previous coordinate (if such exists)
    prev_y : A map from y coordinates to their previous coordinate (if such exists)
    next_x : A map from x coordinates to their next coordinate (if such exists)
    next_y : A map from y coordinates to their next coordinate (if such exists)
    """
    xset: set[float] = set()
    yset: set[float] = set()
    blocks: list[int] = list(range(0, len(carrier.input_problem)))

    for block in carrier.input_problem:
        xset.add(block[0])
        xset.add(block[2])
        yset.add(block[1])
        yset.add(block[3])

    xcoords: list[float] = sorted(xset)
    ycoords: list[float] = sorted(yset)

    next_x: dict[float, float] = {}
    prev_x: dict[float, float] = {}
    next_y: dict[float, float] = {}
    prev_y: dict[float, float] = {}
    for i in range(1, len(xcoords)):
        next_x[xcoords[i - 1]] = xcoords[i]
        prev_x[xcoords[i]] = xcoords[i - 1]
    for i in range(1, len(ycoords)):
        next_y[ycoords[i - 1]] = ycoords[i]
        prev_y[ycoords[i]] = ycoords[i - 1]
    carrier.blocks = blocks
    carrier.prev_x = prev_x
    carrier.prev_y = prev_y
    carrier.next_x = next_x
    carrier.next_y = next_y
    carrier.xcoords = xcoords
    carrier.ycoords = ycoords


def findbestgreedy(carrier: Carrier, ifile, f: float) -> str:
    """
    Calls the greedy algorithm for finding the best solution with just one block
    :param carrier: The inibox variable carrier
    :param ifile: The input file (parsed)
    :param f: The f hyperparameter
    :return: The quality of the solution
    """
    file = open("tofind.txt", "w")
    file.write(getfile(carrier.input_problem, ifile, f))
    file.close()
    run = subprocess.Popen(["./cpp_bin/greedy", "tofind.txt", "bestbox.txt"], stdin=subprocess.PIPE)
    run.wait()
    file = open("bestbox.txt", "r")
    fline: str = file.readline()
    [x1, y1, x2, y2, p] = fline.split()
    inibox: InputBox = (float(x1), float(y1), float(x2), float(y2), float(p))
    print(x1, y1, x2, y2, p)
    carrier.inibox = inibox
    return p


def drawinput(canvas: Canvas, input_problem: InputProblem) -> None:
    """
    Shows an image of the input problem.
    """
    canvas.clear()
    for box in input_problem:
        (a, b, c, d, p) = box
        col = colmix((255, 255, 255), (255, 0, 0), p)
        canvas.drawbox(((a, b), (c, d)), col)
    canvas.show()


def drawoutput(canvas: Canvas, carrier: Carrier, boxes: list[SimpleBox]) -> None:
    """
    Shows an image of a solution.
    """
    canvas.clear()
    for box in carrier.input_problem:
        (a, b, c, d, p) = box
        canvas.drawbox(((a, b), (c, d)))
    for fourbox in boxes:
        (a, b, c, d) = fourbox
        canvas.drawbox(((a, b), (c, d)), "#FF0000")
    canvas.show()


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """
    Main function.
    """
    carrier: Carrier = Carrier()

    canvas = Canvas()

    options = parse_options(prog, args)

    f = options['f']
    file = options['filename']

    carrier.factor = 10000

    ifile = get_ifile(file)

    canvas.setcoords(-1, -1, ifile['Width'] + 1, ifile['Height'] + 1)

    module_names = set()
    if options['module'] is None:
        for r in ifile['Rectangles']:
            for bname in r:
                for lst in r[bname][1]['mod']:
                    for m in lst:
                        module_names.add(m)
    else:
        module_names.add(options['module'])

    allboxes = {}
    for mname in sorted(module_names):
        carrier.input_problem, carrier.selbox = selectbox(mname, ifile)
        if options['plot']:
            drawinput(canvas, carrier.input_problem)

        definecoords(carrier)

        carrier.theoreticalBestArea = 0
        for b in carrier.blocks:
            carrier.theoreticalBestArea += area(carrier, b, True)

        p = findbestgreedy(carrier, ifile, f)

        dif = fstr_to_tuple(carrier, p, f)  # (0, 1)
        print(dif)
        boxes = [(carrier.inibox[0], carrier.inibox[1], carrier.inibox[2], carrier.inibox[3])]
        improvement = True
        for nboxes in [1, 2, 3]:
            print("\n\nnboxes: " + str(nboxes))
            last, tmpb = solve(carrier, ifile, f, dif, nboxes)
            while last[0] > 0:
                boxes = tmpb
                improvement = True
                print("dif = " + str(dif))
                dif = last
                last, tmpb = solve(carrier, ifile, f, dif, nboxes)
            if improvement:
                print(boxes)
                if options['plot']:
                    drawoutput(canvas, carrier, boxes)
                improvement = False
        for i in range(0, len(boxes)):
            (x1, y1, x2, y2) = boxes[i]
            boxes[i] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        allboxes[mname] = boxes
    if options['file'] is not None:
        write_yaml(allboxes, options['file'])
    else:
        print(write_yaml(allboxes))
    return 1


if __name__ == "__main__":
    main()
