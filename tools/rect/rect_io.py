from frame.allocation.allocation import RectAlloc, Allocation

InputBox = tuple[float, float, float, float, float]
InputProblem = list[InputBox]


def getfile(input_problem: InputProblem, ifile, f: float) -> str:
    """
    Outputs the input file for the greedy algorithm
    :param input_problem: The input problem
    :param ifile: The input file (parsed)
    :param f: The f hyperparameter
    :return: The input file for the greedy algorithm, as a string
    """
    outstr = str(ifile['Width']) + " " + str(ifile['Height']) + " " + str(len(input_problem)) + "\n"
    outstr += str(f) + "\n"
    for v in input_problem:
        (x1, y1, x2, y2, p) = v
        outstr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(p) + "\n"
    return outstr


def get_ifile(fname: str):
    test = Allocation(fname)
    obj = {'Width': test.bounding_box.shape.w, 'Height': test.bounding_box.shape.h, 'Rectangles': []}
    for i in range(0, len(test.allocations)):
        b: RectAlloc = test.allocations[i]
        x, y, w, h = b.rect.center.x, b.rect.center.y, b.rect.shape.w, b.rect.shape.h
        item1 = [x, y, w, h]
        item2 = b.alloc
        robj = {'b' + str(i): [{'dim': item1}, {'mod': list(map(lambda q: {q: item2[q]}, item2))}]}
        obj['Rectangles'].append(robj)
    return obj


def selectbox(selbox: str, ifile) -> tuple[InputProblem, str]:
    """
    Receives the module to optimize and preprocesses the problem to leave
    only the relevant information for such module.
    """
    input_problem: InputProblem = []
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
    return input_problem, selbox
