# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from typing import Any
from frame.die.die import Die
from frame.netlist.netlist import Netlist, Module
from argparse import ArgumentParser


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="Verifies that the output netlist has the same relevant "
                                                   "properties as the input netlist", usage='%(prog)s [options]')
    parser.add_argument("ini_netlist", type=str, help="Input netlist (.yaml)")
    parser.add_argument("die", type=str, help="Input die (.yaml)")
    parser.add_argument("out_netlist", type=str, help="Output netlist (.yaml)")
    parser.add_argument("--epsilon", type=float, dest='epsilon', default=1e-10,
                        help="The maximum allowable error")
    return vars(parser.parse_args(args))


def shape_check(m1: Module, m2: Module, epsilon: float):
    def get_hardness(m):
        t = "soft"
        if m.is_hard:
            t = "hard"
        if m.is_fixed:
            t = "fixed"
        return t
    t1 = get_hardness(m1)
    t2 = get_hardness(m2)
    if t1 != t2:
        print("Module", m1.name, "is", t1, "on the input, but", t2, "on the output")
        return False
    if t1 == "hard" or t1 == "fixed":
        a1 = m1.rectangles[0]
        a2 = m2.rectangles[0]
        for i in range(0, m1.num_rectangles):
            r1 = m1.rectangles[i]
            r2 = m2.rectangles[i]
            if abs((r1.center.x - a1.center.x) - (r2.center.x - a2.center.x)) > epsilon or \
               abs((r1.center.y - a1.center.y) - (r2.center.y - a2.center.y)) > epsilon or \
               abs(r1.shape.w - r2.shape.w) > epsilon or abs(r1.shape.h - r2.shape.h) > epsilon:
                print("Module", m1.name, "is hard, but does not keep the same shape")
                return False
        return True
    if t1 == "fixed":
        p1 = m1.rectangles[0].center
        p2 = m2.rectangles[0].center
        if abs(p1.x - p2.x) > epsilon or abs(p1.y - p2.y) > epsilon:
            print("Module", m1.name, "is fixed, but its position changes")
        return True
    return True


def area_check(m1: Module, m2: Module, epsilon: float):
    a1 = m1.area()
    a2 = m2.area()
    if abs(a1 - a2) > epsilon:
        print("Module", m1.name, "has different area on the input and on the output")
        return False
    return True


def die_check(m: Module, die: Die, epsilon: float):
    ok = True
    for rect in m.rectangles:
        x1 = rect.center.x - rect.shape.w / 2
        x2 = rect.center.x + rect.shape.w / 2
        y1 = rect.center.y - rect.shape.h / 2
        y2 = rect.center.y + rect.shape.h / 2
        if max(x1, x2) > die.width + epsilon or \
           min(x1, x2) < -epsilon or \
           max(y1, y2) > die.height + epsilon or \
           min(y1, y2) < -epsilon:
            print("Module", m.name, "falls outside of the die")
            ok = False
    return ok


def rect_overlap(r1, r2, epsilon):
    x1 = r1.center.x - r1.shape.w / 2
    x1b = r1.center.x + r1.shape.w / 2
    y1 = r1.center.y - r1.shape.h / 2
    y1b = r1.center.y + r1.shape.h / 2
    x2 = r2.center.x - r2.shape.w / 2
    x2b = r2.center.x + r2.shape.w / 2
    y2 = r2.center.y - r2.shape.h / 2
    y2b = r2.center.y + r2.shape.h / 2
    left = x2b - epsilon < x1
    right = x1b - epsilon < x2
    bottom = y2b - epsilon < y1
    top = y1b - epsilon < y2
    return not top and not bottom and not left and not right


def overlap_check(m1: Module, m2: Module, epsilon: float):
    for r1 in m1.rectangles:
        for r2 in m2.rectangles:
            if rect_overlap(r1, r2, epsilon):
                print("Modules", m1.name, "and", m2.name, "intersect")
                return False
    return True


def self_overlap_check(m: Module, epsilon: float):
    for r1 in m.rectangles:
        for r2 in m.rectangles:
            if r1 == r2:
                continue
            if rect_overlap(r1, r2, epsilon):
                print("Modules", m.name, "has self-intersecting rectangles")
                return False
    return True


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """
    Main function.
    """
    options = parse_options(prog, args)
    ini_net = Netlist(options['ini_netlist'])
    out_net = Netlist(options['out_netlist'])
    epsilon = options['epsilon']
    die = Die(options['die'])

    o_names = set()
    mod_map = dict()

    ok = True

    for module in ini_net.modules:
        if module.name in mod_map:
            print("Module", module.name, "found twice on the input netlist")
            ok = False
            continue
        mod_map[module.name] = module

    for module in out_net.modules:
        if module.name not in mod_map:
            print("Module", module.name, "is present on the output, but not on the input")
            ok = False
            continue
        if module.name in o_names:
            print("Module", module.name, "found twice on the output netlist")
            ok = False
            continue
        o_names.add(module.name)
        ok &= area_check(mod_map[module.name], module, epsilon)
        ok &= shape_check(mod_map[module.name], module, epsilon)
        ok &= die_check(module, die, epsilon)
        ok &= self_overlap_check(module, epsilon)
        for module2 in out_net.modules:
            if module == module2:
                continue
            ok &= overlap_check(module, module2, epsilon)

    for module in ini_net.modules:
        if module.name not in o_names:
            print("Module", module.name, "is present on the input, but not on the output")
            ok = False

    if ok:
        print("No errors were found!")
    else:
        print("Some errors were found")
    return 0


if __name__ == "__main__":
    main()
