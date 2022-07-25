"""
Package to normalize a fuzzy configuration of
modules in a non-uniform rectangular grid into
adjacent rectangles.
"""

import subprocess
import sys

import pseudobool
import satmanager

from ruamel.yaml import YAML
from canvas import Canvas, colmix

yaml = YAML(typ='safe')

# Custom types
SimpleBox = tuple[float, float, float, float]
InputBox = tuple[float, float, float, float, float]
InputProblem = list[InputBox]
GlobalsType = InputProblem | InputBox | int | str | list[int] | list[float] | dict[float, float] | None


def usage() -> None:
    """
    Prints the usage for this module
    """
    print("Usage:\n python " + sys.argv[0] + " [file name] [options]")
    print("\nOptions:")
    print(" --minarea : Minimizes the total area while guaranteeing a minimum coverage of the original")
    print(" --maxdiff : Maximizes the difference between the inner area and the outer area")
    print(" --minerr  : Minimizes the error (Default)")
    print(" --sf [d]  : Manually set the factor to number d (Not recommended)")
    exit(-1)


def processparam(param: str, i: int, f: float) -> tuple[int, float]:
    """
    Changes the function to optimize in terms of the parameter.
    :param param: The input parameter, of the form --something.
    :param i: The index of the parameter.
    :param f: The current state of the "f" parameter.
    :return: The next values for i and f.
    """
    if param == "--minarea":
        f = 0.89
    elif param == "--maxdiff":
        f = 3.0
    elif param == "--minerr":
        f = 2.0
    elif param == "--sf":
        i = i + 1
        if i >= len(sys.argv):
            usage()
        f = float(sys.argv[i])
    else:
        print("Unknown parameter " + param)
        print()
        usage()
    i = i + 1
    return i, f


def enforce_bb(carrier: dict[str, GlobalsType], ifile, sm: satmanager.SATManager, btag: str, cbtag: str):
    """
    Enforces that a certain rectangle to actually be a rectangle.
    It also enforces the rectangle to be adjacent to the trunk rectangle,
    (if the rectangle is not the trunk rectangle already)
    """
    var_lilx = {}
    var_bigx = {}
    var_lily = {}
    var_bigy = {}
    var_b = {}
    for x in carrier['xcoords']:
        var_lilx[x] = sm.newvar(btag + "x_" + str(x), "")
        var_bigx[x] = sm.newvar(btag + "X_" + str(x), "")
    for y in carrier['ycoords']:
        var_lily[y] = sm.newvar(btag + "y_" + str(y), "")
        var_bigy[y] = sm.newvar(btag + "Y_" + str(y), "")
    for b in carrier['blocks']:
        var_b[b] = sm.newvar(btag + str(b), "")
        sm.imply([var_b[b]], var_lilx[carrier['input_problem'][b][2]])
        sm.imply([var_b[b]], var_bigx[carrier['input_problem'][b][0]])
        sm.imply([var_b[b]], var_lily[carrier['input_problem'][b][3]])
        sm.imply([var_b[b]], var_bigy[carrier['input_problem'][b][1]])
    for i in range(1, len(carrier['xcoords'])):
        sm.imply([var_bigx[carrier['prev_x'][carrier['xcoords'][i]]]], var_bigx[carrier['xcoords'][i]])
        sm.imply([var_lilx[carrier['xcoords'][i]]], var_lilx[carrier['prev_x'][carrier['xcoords'][i]]])
    for i in range(1, len(carrier['ycoords'])):
        sm.imply([var_bigy[carrier['prev_y'][carrier['ycoords'][i]]]], var_bigy[carrier['ycoords'][i]])
        sm.imply([var_lily[carrier['ycoords'][i]]], var_lily[carrier['prev_y'][carrier['ycoords'][i]]])
    for b in carrier['blocks']:
        sm.imply([var_lilx[carrier['next_x'][carrier['input_problem'][b][0]]],
                  var_bigx[carrier['prev_x'][carrier['input_problem'][b][2]]],
                  var_lily[carrier['next_y'][carrier['input_problem'][b][1]]],
                  var_bigy[carrier['prev_y'][carrier['input_problem'][b][3]]]], var_b[b])

    # Enforce connection to the trunk rectangle
    if btag != cbtag:
        north = sm.newvar(btag + "north", "")
        south = sm.newvar(btag + "south", "")
        east = sm.newvar(btag + "east", "")
        west = sm.newvar(btag + "west", "")
        sm.heuleencoding([north, south, east, west])
        sm.pseudoboolencoding(north + south + east + west >= 1)
        for b1 in carrier['blocks']:
            if carrier['input_problem'][b1][0] == 0:
                sm.imply([west], -var_b[b1])
            if carrier['input_problem'][b1][1] == 0:
                sm.imply([north], -var_b[b1])
            if carrier['input_problem'][b1][2] == int(ifile['Width']):
                sm.imply([east], -var_b[b1])
            if carrier['input_problem'][b1][3] == int(ifile['Height']):
                sm.imply([south], -var_b[b1])
            for b2 in carrier['blocks']:
                bb1 = carrier['input_problem'][b1]
                bb2 = carrier['input_problem'][b2]
                if bb1[0] == bb2[2] and bb1[3] > bb2[1] and bb1[1] < bb2[3]:
                    sm.imply([var_b[b1], west, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))
                if bb1[2] == bb2[0] and bb1[3] > bb2[1] and bb1[1] < bb2[3]:
                    sm.imply([var_b[b1], east, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))
                if bb1[1] == bb2[3] and bb1[2] > bb2[0] and bb1[0] < bb2[2]:
                    sm.imply([var_b[b1], north, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))
                if bb1[3] == bb2[1] and bb1[2] > bb2[0] and bb1[0] < bb2[2]:
                    sm.imply([var_b[b1], south, -var_b[b2]], sm.newvar(cbtag + str(b2), ""))


def solve(carrier: dict[str, GlobalsType],
          ifile,
          ratio: float,
          dif: tuple[int, int],
          nboxes: int) -> tuple[tuple[int, int], list[SimpleBox]]:
    """
    Generates the SAT problem, calls the solver and returns the solution.
    :param carrier: The blocks, input_problem, factor and theoreticalBestArea carrier.
    :param ifile: The input file (parsed)
    :param ratio: The f hyperparameter.
    :param dif: The previous optimal solution (as a rational), for optimization purposes.
    :param nboxes: The allowed number of boxes of the solution.
    :return: The cost for an existing solution better than the previous one, and the list of rectangles.
    """
    sm = satmanager.SATManager()

    selarea = pseudobool.Expr()
    vars_b = []
    for b in carrier['blocks']:
        selarea = selarea + area(carrier, b, True) * sm.newvar("b_" + str(b), "")
        vars_b.append(- sm.newvar("b_" + str(b), ""))
        lst = []
        for i in range(0, nboxes):
            sm.imply([sm.newvar("b" + str(i) + "_" + str(b), "")], sm.newvar("b_" + str(b), ""))
            lst.append(- sm.newvar("b" + str(i) + "_" + str(b), ""))
        sm.imply(lst, - sm.newvar("b_" + str(b), ""))
    vars_b.sort(key=lambda x: - carrier['input_problem'][int(x.v[2:])][4])
    realarea = pseudobool.Expr()
    for b in carrier['blocks']:
        realarea = realarea + area(carrier, b, False) * sm.newvar("b_" + str(b), "")
    obj = ratio * selarea - realarea

    # Min area approach
    if ratio < 1:
        sm.pseudoboolencoding(realarea >= round(carrier['theoreticalBestArea'] * ratio))
        sm.pseudoboolencoding(selarea * dif[1] - realarea * dif[0] >= 0)
    # Min error approach
    else:
        sm.pseudoboolencoding(obj >= dif[0])

    for i in range(0, nboxes):
        enforce_bb(carrier, ifile, sm, "b" + str(i) + "_", "b" + str(0) + "_")
    if not sm.solve():
        print("Insat")
        return (0, 1), []
    print("Selected area:    " + str(float(sm.evalexpr(selarea)) / float(carrier['factor'])))
    print("Real area:        " + str(float(sm.evalexpr(realarea)) / float(carrier['factor'])))
    print("Theoretical area: " + str(carrier['theoreticalBestArea'] / float(carrier['factor'])))
    print("Error objective:  " + str(sm.evalexpr(obj)))

    rects = [(float('inf'), float('inf'), -float('inf'), -float('inf'))] * nboxes
    for i in range(0, nboxes):
        for b in carrier['blocks']:
            if sm.value(sm.newvar("b" + str(i) + "_" + str(b), "")) == 1:
                (cx0, cy0, cx1, cy1) = rects[i]
                (nx0, ny0, nx1, ny1, p) = carrier['input_problem'][b]
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


def getfile(carrier: dict[str, GlobalsType], ifile, f: float) -> str:
    """
    Outputs the input file for the greedy algorithm.
    :param carrier: The input_problem variable carrier.
    :param ifile: The input file (parsed).
    :param f: The f hyperparameter.
    :return: The input file for the greedy algorithm, as a string.
    """
    outstr = str(ifile['Width']) + " " + str(ifile['Height']) + " " + str(len(carrier['input_problem'])) + "\n"
    outstr += str(f) + "\n"
    for v in carrier['input_problem']:
        (x1, y1, x2, y2, p) = v
        outstr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(p) + "\n"
    return outstr


def area(carrier: dict[str, GlobalsType], b: int | InputBox, sel: bool) -> float:
    """
    Returns the area of box b.
    :param carrier: The input_problem and factor variable carrier.
    :param b: The given box, either as an index or as a tuple.
    :param sel: Whether we want the total area or just the proportion occupied by the module.
    :return: The area.
    """
    if type(b) is tuple:
        (x1, y1, x2, y2, p) = b
    else:
        (x1, y1, x2, y2, p) = carrier['input_problem'][b]
    if sel:
        return carrier['factor'] * p * (x2 - x1) * (y2 - y1)
    return carrier['factor'] * (x2 - x1) * (y2 - y1)


def fstr_to_tuple(carrier: dict[str, GlobalsType], p: str, f: float) -> tuple[int, int]:
    """
    Turns p, which is a string codifying a float, into a rational.
    :param carrier: The inibox variable carrier.
    :param p: The input string.
    :param f: The f hyperparameter.
    :return: The value of p as a rational.
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
    print(area(carrier, carrier['inibox'], True), area(carrier, carrier['inibox'], False))
    return round(f * area(carrier, carrier['inibox'], True) - area(carrier, carrier['inibox'], False)), 1


def definecoords(carrier: dict[str, GlobalsType]) -> None:
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
    blocks: list[int] = list(range(0, len(carrier['input_problem'])))

    for block in carrier['input_problem']:
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
    carrier['blocks'] = blocks
    carrier['prev_x'] = prev_x
    carrier['prev_y'] = prev_x
    carrier['next_x'] = prev_x
    carrier['next_y'] = prev_x
    carrier['xcoords'] = xcoords
    carrier['ycoords'] = ycoords


def selectbox(carrier: dict[str, GlobalsType], ifile) -> None:
    """
    Receives the module to optimize and preprocesses the problem to leave
    only the relevant information for such module.
    """
    input_problem: InputProblem = []
    selbox = input("Selected module: ")
    for b in ifile['Rectangles']:
        for bname in b:
            [xc, yc, w, h] = b[bname][0]['dim']
            w = float(w)
            h = float(h)
            val = 0.0
            if b[bname][1]['mod'] is not None:
                for i in range(0, len(b[bname][1]['mod'])):
                    if selbox in b[bname][1]['mod'][i]:
                        val = b[bname][1]['mod'][i][selbox]
            input_problem.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, val))
    carrier['input_problem'] = input_problem
    carrier['selbox'] = selbox


def findbestgreedy(carrier: dict[str, GlobalsType], ifile, f: float) -> str:
    """
    Calls the greedy algorithm for finding the best solution with just one block.
    :param carrier: The inibox variable carrier.
    :param ifile: The input file (parsed).
    :param f: The f hyperparameter.
    :return: The quality of the solution.
    """
    file = open("tofind.txt", "w")
    file.write(getfile(carrier, ifile, f))
    file.close()
    run = subprocess.Popen(["greedy", "tofind.txt", "bestbox.txt"], stdin=subprocess.PIPE)
    run.wait()
    file = open("bestbox.txt", "r")
    fline: str = file.readline()
    [x1, y1, x2, y2, p] = fline.split()
    inibox: InputBox = (float(x1), float(y1), float(x2), float(y2), float(p))
    print(x1, y1, x2, y2, p)
    carrier['inibox'] = inibox
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


def drawoutput(canvas: Canvas, carrier: dict[str, GlobalsType], boxes: list[SimpleBox]) -> None:
    """
    Shows an image of a solution.
    """
    canvas.clear()
    for box in carrier['input_problem']:
        (a, b, c, d, p) = box
        canvas.drawbox(((a, b), (c, d)))
    for box in boxes:
        (a, b, c, d) = box
        canvas.drawbox(((a, b), (c, d)), "#FF0000")
    canvas.show()


def main() -> None:
    """
    Main function.
    """
    carrier: dict[GlobalsType] = {}
    carrier['input_problem']: InputProblem
    carrier['selbox']: str
    carrier['factor']: float
    carrier['inibox']: InputBox
    carrier['blocks']: list[int]
    carrier['prev_x']: dict[float, float]
    carrier['prev_y']: dict[float, float]
    carrier['next_x']: dict[float, float]
    carrier['next_y']: dict[float, float]
    carrier['xcoords']: list[float]
    carrier['ycoords']: list[float]
    carrier['theoreticalBestArea']: float

    canvas = Canvas()

    if len(sys.argv) < 2:
        usage()

    f = 2  # 0.89
    i = 1

    while i < len(sys.argv) and sys.argv[i][0:2] == "--":
        i, f = processparam(sys.argv[i], i, f)

    if i >= len(sys.argv):
        usage()

    file = sys.argv[i]
    i = i + 1

    while i < len(sys.argv):
        i, f = processparam(sys.argv[i], i, f)

    carrier['factor'] = 10000

    with open(file, 'r') as file:
        ifile = yaml.load(file)

    canvas.setcoords(-1, -1, ifile['Width'] + 1, ifile['Height'] + 1)

    selectbox(carrier, ifile)
    drawinput(canvas, carrier['input_problem'])

    definecoords(carrier)

    carrier['theoreticalBestArea'] = 0
    for b in carrier['blocks']:
        carrier['theoreticalBestArea'] += area(carrier, b, True)

    p = findbestgreedy(carrier, ifile, f)

    dif = fstr_to_tuple(carrier, p, f)  # (0, 1)
    boxes = [(carrier['inibox'][0], carrier['inibox'][1], carrier['inibox'][2], carrier['inibox'][3])]
    for nboxes in [1, 2, 3]:
        print("\n\nnboxes: " + str(nboxes))
        last, tmpb = solve(carrier, ifile, f, dif, nboxes)
        improvement = False
        while last[0] > 0:
            boxes = tmpb
            improvement = True
            print("dif = " + str(dif))
            dif = last
            last, tmpb = solve(carrier, ifile, f, dif, nboxes)
        if improvement:
            print(boxes)
            drawoutput(canvas, carrier['input_problem'], boxes)


if __name__ == "__main__":
    main()
