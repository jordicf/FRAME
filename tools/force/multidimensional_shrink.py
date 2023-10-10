
from copy import deepcopy
from itertools import combinations
import math
from multiprocessing import Manager, Pool, cpu_count
import random
import numpy as np
from typing import Union
import time

from PIL import Image

from frame.die.die import Die
from frame.geometry.geometry import Point, NPoint
from tools.draw.draw import get_floorplan_plot
from tools.draw.draw import get_graph_plot


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


def vectorized_multidimensional_fruchterman_reingold_layout(
        die: Die, kappa: float = 1.0, num_dimensions: int = 2, verbose: bool = False, visualize: str | None = None,
        max_iter: int = 100, pid: int | None = None, progress = None) -> tuple[Die, list[Image.Image]]:
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

    N = num_dimensions # Number of dimensions (hyperparameter)
    eps = 1e-6

    if verbose:
        print("die.netlist.num_modules:", die.netlist.num_modules)
        print("len(die.netlist.edges):", len(die.netlist.edges))

    assert N >= 2

    draw_method = get_floorplan_plot

    image_idx = 0

    mod2idx = {mod: idx for idx, mod in enumerate(die.netlist.modules)}

    sides = np.array([die.width, die.height] + [max(die.width, die.height)] * (N-2))
    dsides = sides / (max_iter+1)
    dsides[0] = 0
    dsides[1] = 0

    t = max(die.width, die.height) * 0.1
    dt = t / (max_iter + 1)

    k = kappa * (die.width * die.height / die.netlist.num_modules)**(1 / 2)
    scale = np.array([1 if i < 2 else 0.9 for i in range(N)])

    def f_att(x, w):
        return w * x**2 / k

    def f_rep(x, w):
        return w * k**2 / np.maximum(x, 1e-6)

    def norm(x):
        return np.sqrt(np.dot(x, x))

    pos: list[NPoint] = [NPoint(module.center, n=N) - NPoint([die.width, die.height], n=N) / 2 if module.center is not None else NPoint(n=N)
                        for module in die.netlist.modules]  # The die is recentered to the origin

    disp = [NPoint(n=N)] * die.netlist.num_modules

    if visualize is not None:
        for v, module in enumerate(die.netlist.modules):
            module.center = Point(*pos[v][:2]) + Point(die.width, die.height) / 2
            if module.is_hard and not module.is_fixed:
                module.recenter_rectangles()
        get_graph_plot(die.netlist, f"/tmp/FRAME-tmp-image-{(image_idx:=image_idx+1)}")

    for it in range(max_iter):
        if pid is not None:
            progress.value = it

        pos_mat = np.array([e.x for e in pos]).T

        # Compute repulsion
        for v in range(die.netlist.num_modules):
            diff_mat = np.array([pos[v].x] * len(pos)).T - pos_mat
            area_v = die.netlist.modules[v].area()
            die_repelling_v = NPoint(n=N)
            diff_norm_vec = np.maximum(
                np.apply_along_axis(norm, 0, diff_mat),
                1e-6
            )
            disp_mat = ((diff_mat / diff_norm_vec * f_rep(diff_norm_vec, area_v)).T + die_repelling_v.x).T
            disp_mat[:,v] = 0
            disp[v] = NPoint(np.sum(disp_mat, axis=1))

        # Compute attraction
        for hyperedge in die.netlist.edges:
            for v_mod, u_mod in combinations(hyperedge.modules, 2):
                v, u = mod2idx[v_mod], mod2idx[u_mod]
                diff = pos[v] - pos[u]
                diff_norm = max(diff.norm(), 1e-6)
                disp[v] -= diff / diff_norm * f_att(diff_norm, hyperedge.weight)
                disp[u] += diff / diff_norm * f_att(diff_norm, hyperedge.weight) # TODO: edge or node weight??

        # Apply computed forces and contract space
        for v in range(die.netlist.num_modules):
            if not die.netlist.modules[v].is_fixed:
                disp_norm = max(disp[v].norm(), 1e-6)
                pos[v] += disp[v] / disp_norm * min(disp_norm, t) * NPoint([1, 1] + [(1-it/max_iter)]*(N-2))
                pos[v] *= NPoint([1, 1] + [(1-it/max_iter)]*(N-2))

        # Reduce temperatures and dimensions scale
        t -= dt
        sides -= dsides

        if verbose:
            print(it, end=" ", flush=True)
        if visualize is not None:
            for v, module in enumerate(die.netlist.modules):
                module.center = Point(*pos[v][:2])
                if module.is_hard and not module.is_fixed:
                    module.recenter_rectangles()
            get_graph_plot(die.netlist, f"/tmp/FRAME-tmp-image-{(image_idx:=image_idx+1)}") 

    if verbose:
        print("\nAlgorithm completed.")

    if visualize is None:
        for v, module in enumerate(die.netlist.modules):
            module.center = Point(*pos[v][:2]) + Point(die.width, die.height) / 2
            if module.is_hard and not module.is_fixed:
                module.recenter_rectangles()

    return die, []


def mp_function_wrapper(args):
    # usage: mp_function_wrapper with args = (function, function args...)
    return args[0](*args[1:])


def force_algorithm(die: Die, verbose: bool = False, visualize: str | None = None, max_iter: int = 100, 
                    num_dimensions: int = 2, parallelize: bool = False) \
        -> tuple[Die, list[Image.Image]]:
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
    kappas = [2**k/100 for k in range(8)] # TODO: choose better kappas
    kappa_result = {}

    if verbose:
        print("kappas:", kappas)

    if parallelize:
        from tqdm import tqdm
        
        # Execute with multiple kappas simultaneously using parallelization
        manager = Manager()
        progress = [manager.Value('i', 0) for _ in range(len(kappas))]
        pool = Pool(min(len(kappas), cpu_count()))
        with pool:
            results = pool.map_async(mp_function_wrapper, [
                    [
                        vectorized_multidimensional_fruchterman_reingold_layout, 
                        deepcopy(die), 
                        kappas[i], 
                        num_dimensions, 
                        False, 
                        None,
                        max_iter,
                        i,
                        progress[i]
                    ] for i in range(len(kappas))
                ]
            )

            # Show progress of the multiple kappas execution
            total = max_iter*len(kappas)
            if verbose:
                t = tqdm(total=total)
                last_progress = 0
            while not results.ready():
                if verbose:
                    current_progress = 0
                    for val in progress:
                        current_progress += val.value
                    t.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(0.1)
            if verbose:
                t.update(total - last_progress)
                t.close()

            results = results.get()

            for i in range(len(results)):
                new_die, _ = results[i]
                kappa = kappas[i]
                kappa_result[kappa] = results[i]
                assert new_die.netlist is not None  # Assertion to suppress Mypy error
                intersection_area = total_intersection_area(new_die)
                wire_length = new_die.netlist.wire_length
                cost = intersection_area + wire_length / 2
                if verbose:
                    print(f"Kappa: {kappa}")
                    print(f"Intersection area: {intersection_area} | Wire length: {wire_length} | Cost: {cost}")
                    print("--------------------")
                if cost < best_cost:
                    best_cost = cost
                    best_kappa = kappa
    else:
        for kappa in kappas:
            if verbose:
                print(f"Kappa: {kappa}")
            kappa_result[kappa] = vectorized_multidimensional_fruchterman_reingold_layout(deepcopy(die), kappa, num_dimensions, verbose, None, max_iter)
            new_die, _ = kappa_result[kappa]
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

    if verbose:
        print(f"Best cost (1): {best_cost}")
    if visualize:
        if verbose:
            print("Recalculating layout with best kappa", end="")
            if visualize:
                print(" and creating the visualization", end="")
            print("...")
        return vectorized_multidimensional_fruchterman_reingold_layout(deepcopy(die), best_kappa, num_dimensions, verbose, visualize, max_iter)
    else:
        return kappa_result[best_kappa]
