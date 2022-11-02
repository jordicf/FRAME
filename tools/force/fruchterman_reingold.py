# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from copy import deepcopy
from itertools import combinations
import math

from frame.die.die import Die
from frame.geometry.geometry import Point
from tools.draw.draw import get_floorplan_plot


def circle_circle_intersection_area(c1: Point, r1: float, c2: Point, r2: float) -> float:
    """
    Computes the area of the intersection of two circles
    :param c1: center of the first circle
    :param r1: radius of the first circle
    :param c2: center of the second circle
    :param r2: radius of the second circle
    :return: the area of the intersection of the two circles
    """
    d = (c1 - c2).norm()
    if d > r1 + r2:
        return 0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2)**2
    alpha = math.acos((r1**2 + d**2 - r2**2) / (2 * r1 * d))
    beta = math.acos((r2**2 + d**2 - r1**2) / (2 * r2 * d))
    return r1**2 * alpha + r2**2 * beta - d * r1 * math.sin(alpha)


def total_intersection_area(die: Die) -> float:
    """
    Computes the total area of the intersection of all the circles in the die
    :param die: the die, with the netlist with the modules with the centers initialized
    :return: the total area of the intersection of all the circles in the die
    """
    assert die.netlist is not None  # Assertion to suppress Mypy error
    area = 0.0
    for m1 in die.netlist.modules:
        for m2 in die.netlist.modules:
            if m1 != m2:
                assert m1.center is not None and m2.center is not None  # Assertion to suppress Mypy error
                area += circle_circle_intersection_area(m1.center, math.sqrt(m1.area() / math.pi),
                                                        m2.center, math.sqrt(m2.area() / math.pi))
    return area


def fruchterman_reingold_layout(die: Die, kappa: float = 1.0, verbose: bool = False, visualize: str | None = None,
                                max_iter: int = 100) -> Die:
    """
    Computes a layout for the die netlist using a modified version of the Fruchterman-Reingold algorithm
    :param die: the die, with the netlist with the modules with the centers initialized
    :param kappa: hyperparameter that multiplies the spring constant k used in the algorithm
    :param verbose: if True, prints information to the console
    :param visualize: if not None, saves the intermediate layouts as a GIF with the given name
    :param max_iter: the maximum number of iterations of the algorithm
    :return: the die with the netlist with the center of the modules updated to the computed layout
    """
    assert die.netlist is not None, "No netlist associated to the die"

    vis_imgs = []

    mod2idx = {mod: idx for idx, mod in enumerate(die.netlist.modules)}

    t = max(die.width, die.height) * 0.1
    dt = t / (max_iter + 1)

    k = kappa * (die.width * die.height / die.netlist.num_modules)**(1 / 2)

    def f_att(x, w):
        return w * x**2 / k

    def f_rep(x, w):
        return w * k**2 / max(x, 1e-6)

    def die_repelling(p: Point, w: float) -> Point:
        repelling = Point(0, 0)
        if p.x < -die.width / 2 + die.width / 10:
            repelling += Point(1, 0) * f_rep(p.x + die.width / 2, w)
        if p.x > die.width / 2 - die.width / 10:
            repelling += Point(-1, 0) * f_rep(die.width / 2 - p.x, w)
        if p.y < -die.height / 2 + die.height / 10:
            repelling += Point(0, 1) * f_rep(p.y + die.height / 2, w)
        if p.y > die.height / 2 - die.height / 10:
            repelling += Point(0, -1) * f_rep(die.height / 2 - p.y, w)
        return repelling

    pos: list[Point] = [module.center - Point(die.width, die.height) / 2 if module.center is not None else Point()
                        for module in die.netlist.modules]  # The die is recentered to the origin
    disp = [Point()] * die.netlist.num_modules

    if visualize is not None:
        for v, module in enumerate(die.netlist.modules):
            module.center = pos[v] + Point(die.width, die.height) / 2
        vis_imgs.append(get_floorplan_plot(die.netlist, die.bounding_box.shape))

    for i in range(max_iter):
        for v in range(die.netlist.num_modules):
            disp[v] = Point()
            for u in range(die.netlist.num_modules):
                if u != v:
                    diff = pos[v] - pos[u]
                    diff_norm = max(diff.norm(), 1e-6)
                    disp[v] += diff / diff_norm * f_rep(diff_norm, die.netlist.modules[v].area()) \
                        + die_repelling(pos[v], die.netlist.modules[v].area())

        for hyperedge in die.netlist.edges:
            for v_mod, u_mod in combinations(hyperedge.modules, 2):
                v, u = mod2idx[v_mod], mod2idx[u_mod]
                diff = pos[v] - pos[u]
                diff_norm = max(diff.norm(), 1e-6)
                disp[v] -= diff / diff_norm * f_att(diff_norm, hyperedge.weight)
                disp[u] += diff / diff_norm * f_att(diff_norm, hyperedge.weight)

        for v in range(die.netlist.num_modules):
            if not die.netlist.modules[v].is_fixed:
                disp_norm = max(disp[v].norm(), 1e-6)
                pos[v] += disp[v] / disp_norm * min(disp_norm, t)
                pos[v].x = min(die.width / 2, max(-die.width / 2, pos[v].x))
                pos[v].y = min(die.height / 2, max(-die.height / 2, pos[v].y))

        t -= dt

        if verbose:
            print(i, end=" ", flush=True)

        if visualize is not None:
            for v, module in enumerate(die.netlist.modules):
                module.center = pos[v] + Point(die.width, die.height) / 2
            vis_imgs.append(get_floorplan_plot(die.netlist, die.bounding_box.shape))

    if verbose:
        print("\nAlgorithm completed.")

    if visualize is None:
        for v, module in enumerate(die.netlist.modules):
            module.center = pos[v] + Point(die.width, die.height) / 2
    else:
        vis_imgs[0].save(f"{visualize}.gif", save_all=True, append_images=vis_imgs[1:], duration=100)

    return die


def force_algorithm(die: Die, verbose: bool = False, visualize: str | None = None, max_iter: int = 100) -> Die:
    """
    Computes multiple layouts for the die netlist by changing the kappa hyperparameter, and returns the one with the
    smallest cost, which is defined as the sum of the areas of the intersections of the circles of the modules plus half
    the netlist wire length. The layout algorithm is based on the Fruchterman-Reingold algorithm
    :param die: the die, with the netlist with the modules with the centers initialized
    :param verbose: if True, prints information to the console
    :param visualize: if not None, saves the intermediate layouts as a GIF with the given name
    :param max_iter: the maximum number of iterations of the algorithm
    :return: the die with the netlist with the center of the modules updated to the computed layout
    """
    best_cost = float("inf")
    best_kappa = 0.0
    for kappa in [i / 10 for i in range(4, 16)]:
        if verbose:
            print(f"Kappa: {kappa}")
        new_die = fruchterman_reingold_layout(deepcopy(die), kappa, verbose, None, max_iter)
        assert new_die.netlist is not None  # Assertion to suppress Mypy error
        intersection_area = total_intersection_area(new_die)
        wire_length = new_die.netlist.wire_length
        cost = intersection_area + wire_length / 2
        if verbose:
            print(f"Intersection area: {intersection_area} | Wire length: {wire_length} | Cost: {cost}")
            print("--------------------")
        if cost < best_cost:
            best_cost = cost
            best_kappa = kappa

    if verbose:
        print(f"Best kappa: {best_kappa}")
        print("Recalculating layout with best kappa", end="")
        if visualize:
            print(" and creating the visualization", end="")
        print("...")

    return fruchterman_reingold_layout(die, best_kappa, verbose, visualize, max_iter)

