"""
Package to normalize a fuzzy configuration of
modules in a non-uniform rectangular grid into
adjacent rectangles.
"""

import subprocess
import sys

import pseudobool
import satmanager

import yaml

def usage():
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

def processParam(param: str, i: int, f: float):
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

def solve(ratio, dif, nboxes):
    """
    Generates the SAT problem, calls the solver and returns the solution.
    TODO: Split this function further down into smaller functions: The function is too long.
    :param ratio: The f hyperparameter.
    :param dif: The previous optimal solution, for optimization purposes.
    :param nboxes: The allowed number of boxes of the solution.
    :return: The cost for an existing solution better than the previous one.
    """
    sm = satmanager.SATManager()

    def enforce_bb(btag, cbtag):
        var_x = {}
        var_X = {}
        var_y = {}
        var_Y = {}
        var_b = {}
        for x in xcoords:
            var_x[x] = sm.newVar(btag + "x_" + str(x), "")
            var_X[x] = sm.newVar(btag + "X_" + str(x), "")
        for y in ycoords:
            var_y[y] = sm.newVar(btag + "y_" + str(y), "")
            var_Y[y] = sm.newVar(btag + "Y_" + str(y), "")
        for b in blocks:
            var_b[b] = sm.newVar(btag + str(b), "")
            sm.imply([var_b[b]], var_x[input_problem[b][2]])
            sm.imply([var_b[b]], var_X[input_problem[b][0]])
            sm.imply([var_b[b]], var_y[input_problem[b][3]])
            sm.imply([var_b[b]], var_Y[input_problem[b][1]])
        for i in range(1, len(xcoords)):
            sm.imply([var_X[prev_x[xcoords[i]]]], var_X[xcoords[i]])
            sm.imply([var_x[xcoords[i]]], var_x[prev_x[xcoords[i]]])
        for i in range(1, len(ycoords)):
            sm.imply([var_Y[prev_y[ycoords[i]]]], var_Y[ycoords[i]])
            sm.imply([var_y[ycoords[i]]], var_y[prev_y[ycoords[i]]])
        for b in blocks:
            sm.imply([var_x[next_x[input_problem[b][0]]],
                      var_X[prev_x[input_problem[b][2]]],
                      var_y[next_y[input_problem[b][1]]],
                      var_Y[prev_y[input_problem[b][3]]]], var_b[b])
        if btag != cbtag:
            north = sm.newVar(btag + "north", "")
            south = sm.newVar(btag + "south", "")
            east = sm.newVar(btag + "east", "")
            west = sm.newVar(btag + "west", "")
            sm.heuleEncoding([north, south, east, west])
            sm.pseudoboolEncoding(north + south + east + west >= 1)
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
                    B1 = input_problem[b1]
                    B2 = input_problem[b2]
                    if B1[0] == B2[2] and B1[3] > B2[1] and B1[1] < B2[3]:
                        sm.imply([var_b[b1], west, -var_b[b2]], sm.newVar(cbtag + str(b2), ""))
                    if B1[2] == B2[0] and B1[3] > B2[1] and B1[1] < B2[3]:
                        sm.imply([var_b[b1], east, -var_b[b2]], sm.newVar(cbtag + str(b2), ""))
                    if B1[1] == B2[3] and B1[2] > B2[0] and B1[0] < B2[2]:
                        sm.imply([var_b[b1], north, -var_b[b2]], sm.newVar(cbtag + str(b2), ""))
                    if B1[3] == B2[1] and B1[2] > B2[0] and B1[0] < B2[2]:
                        sm.imply([var_b[b1], south, -var_b[b2]], sm.newVar(cbtag + str(b2), ""))

    selarea = pseudobool.Expr()
    vars_b = []
    for b in blocks:
        selarea = selarea + area(b, True) * sm.newVar("b_" + str(b), "")
        vars_b.append(- sm.newVar("b_" + str(b), ""))
        l = []
        for i in range(0, nboxes):
            sm.imply([sm.newVar("b" + str(i) + "_" + str(b), "")], sm.newVar("b_" + str(b), ""))
            l.append(- sm.newVar("b" + str(i) + "_" + str(b), ""))
        sm.imply(l, - sm.newVar("b_" + str(b), ""))
    vars_b.sort(key=lambda x: - input_problem[int(x.v[2:])][4])
    realarea = pseudobool.Expr()
    for b in blocks:
        realarea = realarea + area(b, False) * sm.newVar("b_" + str(b), "")
    obj = ratio * selarea - realarea
    # Min area approach
    if ratio < 1:
        sm.pseudoboolEncoding(realarea >= round(theoreticalBestArea * ratio))
        sm.pseudoboolEncoding(selarea * dif[1] - realarea * dif[0] >= 0)
    # Min error approach
    else:
        sm.pseudoboolEncoding(obj >= dif[0])
    for i in range(0, nboxes):
        enforce_bb("b" + str(i) + "_", "b" + str(0) + "_")
    if not sm.solve():
        print("Insat")
        return (0, 1)
    for j in range(0, ifile['Height']):
        s = ""
        for i in range(0, ifile['Width']):
            b = bmap[i + j * ifile['Width']]
            if sm.value(sm.newVar("b_" + str(b), "")) == 1:
                s += symbols[selbox]
            else:
                s += "."
        print(s)
    print("Selected area:    " + str(float(sm.evalExpr(selarea)) / float(factor)))
    print("Real area:        " + str(float(sm.evalExpr(realarea)) / float(factor)))
    print("Theoretical area: " + str(theoreticalBestArea / float(factor)))
    print("Error objective:  " + str(sm.evalExpr(obj)))
    # Min area approach
    if ratio < 1:
        return (sm.evalExpr(selarea) + 1, sm.evalExpr(realarea))
    # Min error approach
    else:
        return (sm.evalExpr(obj) + 1, 1)

def getFile(f: float) -> str:
    """
    Outputs the input file for the greedy algorithm.
    :param f: The f hyperparameter.
    :return: The input file for the greedy algorithm, as a string.
    """
    outstr = str(ifile['Width']) + " " + str(ifile['Height']) + " " + str(len(input_problem)) + "\n"
    outstr += str(f) + "\n"
    for v in input_problem:
        (x1, y1, x2, y2, p) = v
        outstr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(p) + "\n"
    return outstr

def area(b, sel: bool) -> float:
    """
    Returns the area of box b.
    :param b: The given box, either as an index or as a tuple.
    :sel: Whether we want the total area or just the proportion occupied by the module.
    :return: The area.
    """
    if type(b) is tuple:
        (x1, y1, x2, y2, p) = b
    else:
        (x1, y1, x2, y2, p) = input_problem[b]
    if sel:
        return factor * p * (x2 - x1) * (y2 - y1)
    return factor * (x2 - x1) * (y2 - y1)

def fstr_to_tuple(p: str, f):
    """
    Turns p, which is a string codifying a float, into a rational.
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
        return (n1, n2)
    # Min error approach
    print(area(inibox, True), area(inibox, False))
    return (round(f * area(inibox, True) - area(inibox, False)), 1)

def main():
    """
    Main function.
    TODO: Split the function further down into smaller functions.
    This is too long for a main function.
    """
    
    if len(sys.argv) < 2:
        usage()
    
    f = 2  # 0.89
    i = 1
    
    while i < len(sys.argv) and sys.argv[i][0:2] == "--":
        i, f = processParam(sys.argv[i])
    
    if i >= len(sys.argv):
        usage()
    
    file = sys.argv[i]
    i = i + 1
    
    while i < len(sys.argv):
        i, f = processParam(sys.argv[i])
    
    factor = 10000
    symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    
    ifile = None
    with open(file, 'r') as file:
        ifile = yaml.load(file, Loader=yaml.FullLoader)
    
    bmap = [0] * (ifile['Width'] * ifile['Height'])
    input_problem = []
    selbox = int(input("Selected module: "))
    for b in ifile['Rectangles']:
        for bname in b:
            [x1, y1, x2, y2, l] = b[bname]
            input_problem.append((x1, y1, x2, y2, l[selbox]))
            for i in range(x1, x2):
                for j in range(y1, y2):
                    bmap[i + j * ifile['Width']] = len(input_problem) - 1
    
    xset = set()
    yset = set()
    blocks = range(0, len(input_problem))
    
    for block in input_problem:
        xset.add(block[0])
        xset.add(block[2])
        yset.add(block[1])
        yset.add(block[3])
    
    xcoords = sorted(xset)
    ycoords = sorted(yset)
    
    next_x = {}
    prev_x = {}
    next_y = {}
    prev_y = {}
    for i in range(1, len(xcoords)):
        next_x[xcoords[i - 1]] = xcoords[i]
        prev_x[xcoords[i]] = xcoords[i - 1]
    for i in range(1, len(ycoords)):
        next_y[ycoords[i - 1]] = ycoords[i]
        prev_y[ycoords[i]] = ycoords[i - 1]
    
    theoreticalBestArea = 0
    for b in blocks:
        theoreticalBestArea += area(b, True)
    
    file = open("tofind.txt", "w")
    file.write(getFile(f))
    file.close()
    run = subprocess.Popen(["greedy", "tofind.txt", "bestbox.txt"], stdin=subprocess.PIPE)
    run.wait()
    file = open("bestbox.txt", "r")
    fline = file.readline()
    [x1, y1, x2, y2, p] = fline.split()
    inibox = (int(x1), int(y1), int(x2), int(y2), float(p))
    print(x1, y1, x2, y2, p)
    
    dif = fstr_to_tuple(p, f)  # (0, 1)
    for nboxes in [1, 2, 3]:
        print("\n\nnboxes: " + str(nboxes))
        last = solve(f, dif, nboxes)
        while last[0] > 0:
            print("dif = " + str(dif))
            dif = last
            last = solve(f, dif, nboxes)


if __name__ == "__main__":
    main()
