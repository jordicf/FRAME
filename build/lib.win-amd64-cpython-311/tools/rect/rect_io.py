# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).


from frame.allocation.allocation import RectAlloc, Allocation
from frame.geometry.geometry import Point
from frame.netlist.netlist import Netlist

InputBox = tuple[float, float, float, float, float]
SimpleBox = tuple[float, float, float, float]
InputProblem = list[InputBox]
Dims = list[float]
Module = dict[str, float]
Mods = list[Module]
IFileTerm = dict[str, Dims | Mods]
IFileBox = dict[str, list[IFileTerm]]


def getfile(input_problem: InputProblem, ifile, f: float) -> str:
    """
    Outputs the input file for the greedy algorithm
    :param input_problem: The input problem
    :param ifile: The input file (parsed)
    :param f: The f hyperparameter
    :return: The input file for the greedy algorithm, as a string
    """
    output_string = str(ifile['Width']) + " " + str(ifile['Height']) + " " + str(len(input_problem)) + "\n"
    output_string += str(f) + "\n"
    for v in input_problem:
        (x1, y1, x2, y2, p) = v
        output_string += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(p) + "\n"
    return output_string


def get_alloc(file_name: str):
    test = Allocation(file_name)
    obj = {'Width': test.bounding_box.shape.w, 'Height': test.bounding_box.shape.h, 'Rectangles': []}
    rectangle_list: list[IFileBox] = []
    for i in range(0, len(test.allocations)):
        b: RectAlloc = test.allocations[i]
        x, y, w, h = b.rect.center.x, b.rect.center.y, b.rect.shape.w, b.rect.shape.h
        item1 = [x, y, w, h]
        item2 = b.alloc
        dim: IFileTerm = {'dim': item1}
        mod: IFileTerm = {'mod': list(map(lambda q: {q: item2[q]}, item2))}
        lst: list[IFileTerm] = [dim, mod]
        rob: IFileBox = {f"b{str(i)}": lst}
        rectangle_list.append(rob)
    obj['Rectangles'] = rectangle_list
    return obj


def get_netlist(file_name: str, allocation_name: str):
    alloc = Allocation(allocation_name)
    if file_name is None:
        module_map: dict[str, tuple[Point, float]] = {}
        for i in range(0, len(alloc.allocations)):
            b: RectAlloc = alloc.allocations[i]
            for m in b.alloc:
                if m not in module_map:
                    module_map[m] = (b.rect.center, b.rect.area * b.alloc[m])
                else:
                    (c1, a1) = module_map[m]
                    c2, a2 = b.rect.center, b.rect.area * b.alloc[m]
                    module_map[m] = (c1 * (a1 / (a1 + a2)) + c2 * (a2 / (a1 + a2)), a1 + a2)

        netlist_string = "Modules: {"
        first = True
        for key in module_map:
            if not first:
                netlist_string += ","
            first = False
            netlist_string += "\n  " + key + ": {\n    area: " + str(module_map[key][1]) + ",\n    "
            netlist_string += "center: [" + str(module_map[key][0].x) + ", " + str(module_map[key][0].y) + "]\n"
            netlist_string += "  }"
        netlist_string += "\n}\n\nNets: []\n"
        return Netlist(netlist_string)
    return Netlist(file_name)


def select_box(selected_box: str, ifile) -> tuple[InputProblem, str]:
    """
    Receives the module to optimize and preprocesses the problem to leave
    only the relevant information for such module.
    """
    input_problem: InputProblem = []
    for b in ifile['Rectangles']:
        for box_name in b:
            [xc, yc, w, h] = b[box_name][0]['dim']
            w = float(w)
            h = float(h)
            val = 0.0
            if b[box_name][1]['mod'] is not None:
                for i in range(0, len(b[box_name][1]['mod'])):
                    if selected_box in b[box_name][1]['mod'][i]:
                        val = b[box_name][1]['mod'][i][selected_box]
            input_problem.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, val))
    return input_problem, selected_box


def solution_to_netlist(netlist: Netlist, result: dict[str, list[SimpleBox]]):
    netlist_string = "Modules: {"
    first = True
    for module in netlist.modules:
        if not first:
            netlist_string += ","
        first = False
        netlist_string += "\n  " + module.name + ": {\n    "
        if module.name in result:
            netlist_string += "rectangles: " + str(list(map(lambda x: [x[0], x[1], x[2], x[3]], result[module.name])))
        elif len(module.rectangles) > 0:
            netlist_string += "rectangles: " + str(list(map(lambda x: [x.center.x, x.center.y, x.shape.w, x.shape.h],
                                                            module.rectangles))) + ""
        else:
            point = module.center
            if isinstance(point, Point):
                netlist_string += "center: [" + str(point.x) + ", " + str(point.y) + "]"
            else:
                raise Exception("I don't know what to do with module " + str(module.name))
        if not module.is_hard:
            netlist_string += ",\n    area: " + str(module.area())
        if module.is_fixed:
            netlist_string += ",\n    fixed: true"
        elif module.is_hard:
            netlist_string += ",\n    hard: true"
        netlist_string += "\n  }"
    netlist_string += "}\n\nNets: ["
    first = True
    if len(netlist.edges) == 0:
        netlist_string += "]\n"
    else:
        for hyper_edge in netlist.edges:
            if not first:
                netlist_string += ","
            first = False
            netlist_string += "\n  ["
            f2 = True
            for module in hyper_edge.modules:
                if not f2:
                    netlist_string += ", "
                f2 = False
                netlist_string += module.name
            netlist_string += "]"
        netlist_string += "\n]\n"
    return netlist_string
