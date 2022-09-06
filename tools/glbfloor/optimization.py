# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""This file contains the code for optimizing the floor plans"""
from dataclasses import astuple
from typing import Callable, Union

from PIL import Image
from gekko import GEKKO
from gekko.gk_variable import GKVariable

from frame.geometry.geometry import Point, Rectangle
from frame.die.die import Die
from frame.allocation.allocation import AllocDescriptor, Alloc, Allocation, create_initial_allocation
from frame.netlist.module import Module

from tools.draw.draw import get_floorplan_plot as get_joint_floorplan_plot
from tools.glbfloor.plots import PlottingOptions, get_separated_floorplan_plot

GEKKOType = Union[float, GKVariable]


class Model:
    """GEKKO model with variables"""
    gekko: GEKKO

    # Model variables
    x: dict[str, GEKKOType]
    y: dict[str, GEKKOType]
    d: dict[str, GEKKOType]
    a: dict[str, dict[int, GEKKOType]]

    def __init__(self):
        """Constructs the GEKKO object"""
        self.gekko = GEKKO(remote=False)
        self.x = {}
        self.y = {}
        self.d = {}
        self.a = {}


def get_value(v) -> float:
    """
    Get the value of the GEKKO object v
    :param v: a variable or a value
    :return: the value of v
    """
    if not isinstance(v, float):
        v = v.value.value
        if hasattr(v, "__getitem__"):
            v = v[0]
    if not isinstance(v, float):
        try:
            v = float(v)
        except TypeError:
            raise ValueError(f"Could not get value of {v} (type: {type(v)}")
    return v


DispersionFunction = Callable[[GEKKOType, GEKKOType], GEKKOType]


def calculate_dispersions(modules: list[Module], allocation: Allocation, dispersion_function: DispersionFunction) \
        -> dict[str, float]:
    """
    Calculate the dispersions of the modules
    :param modules: modules with centroids initialized
    :param allocation: the allocation of the modules
    :param dispersion_function: the function to use to calculate the dispersion of each module
    :return: a dictionary from module name to float pair which indicates the dispersion of each module in the
    given netlist and allocation
    """
    dispersions = {}
    for module in modules:
        m = module.name
        assert module.center is not None
        dispersions[m] = 0.0
        for module_alloc in allocation.allocation_module(m):
            cell = allocation.allocation_rectangle(module_alloc.rect_index).rect
            area = cell.area * module_alloc.area_ratio
            dispersions[m] += area * dispersion_function(module.center.x - cell.center.x,
                                                         module.center.y - cell.center.y)

    return dispersions


def get_neighbouring_cells(allocation: Allocation, cell_index: int) -> list[int]:
    """
    Given an allocation and a cell index, returns a list of the indices of the cells neighbouring the specified cell
    :param allocation: the allocation
    :param cell_index: the index of the cell
    :return: the list of indices of the neighbouring cells
    """
    n_cells = allocation.num_rectangles
    neigh_cells = []
    for c in range(n_cells):
        if c != cell_index and \
                allocation.allocation_rectangle(cell_index).rect.touches(allocation.allocation_rectangle(c).rect):
            neigh_cells.append(c)
    return neigh_cells


def extract_solution(model: Model, die: Die, cells: list[Rectangle], threshold: float) \
        -> tuple[Die, Allocation, dict[str, float]]:
    """
    Extracts the solution from the model
    :param model: the model
    :param die: die with netlist containing the modules with centroids initialized
    :param cells: cells of the floor plan to allocate the modules
    :param threshold: hyperparameter between 0 and 1 to decide if allocations can be fixed
    :return:
    - die - die with netlist with the centroids of the modules updated
    - allocation - optimized allocation
    - dispersions - a dictionary which indicates the dispersion of each module
    """
    assert die.netlist is not None, "No netlist associated to the die"

    allocation_list: list[AllocDescriptor] = []
    for c, cell in enumerate(cells):
        alloc = Alloc()
        for module in die.netlist.modules:
            m = module.name
            a_mc_val = get_value(model.a[m][c])
            if a_mc_val > 1 - threshold:
                alloc[m] = a_mc_val
        if alloc:  # add only if not empty
            allocation_list.append((cell, alloc, 0))
    allocation = Allocation(allocation_list)

    dispersions = {}
    for module in die.netlist.modules:
        m = module.name
        module.center = Point(get_value(model.x[m]), get_value(model.y[m]))
        dispersions[m] = get_value(model.d[m]) if m in model.d else 0.0

    return die, allocation, dispersions


def solve_and_extract_solution(model: Model, die: Die, cells: list[Rectangle], threshold: float, max_iter: int = 100,
                               verbose: bool = False, plotting_options: PlottingOptions | None = None) \
        -> tuple[Die, Allocation, dict[str, float], tuple[list[Image.Image], list[Image.Image]]]:
    """
    Solves the model's optimization problem, extracts the solution from it, and returns it
    :param model: the model
    :param die: die with netlist containing the modules with centroids initialized
    :param cells: cells of the floor plan to allocate the modules
    :param threshold: hyperparameter between 0 and 1 to decide if allocations can be fixed
    :param max_iter: maximum number of iterations for GEKKO
    :param verbose: if True, the GEKKO optimization log is displayed (not supported if visualize is True)
    :param plotting_options: plotting options
    :return:
    - die - die with netlist with the centroids of the modules updated
    - dispersions - a dictionary which indicates the dispersion of each module
    - vis_imgs - a tuple of two lists of images visualizing the optimization containing joint and separated plots,
    respectively. If those options are False, the respective list is empty. If visualize is False, a tuple of two empty
    lists is returned.
    """
    allocation = None
    dispersions = None
    vis_imgs: tuple[list[Image.Image], list[Image.Image]] = ([], [])
    if plotting_options is None or not plotting_options.visualize:  # if not visualize
        model.gekko.options.MAX_ITER = max_iter
        model.gekko.solve(disp=verbose)
        die, allocation, dispersions = extract_solution(model, die, cells, threshold)
    else:
        # See https://stackoverflow.com/a/73196238/10152624 for the method used here
        i = 0
        while i < max_iter:
            model.gekko.options.MAX_ITER = i
            model.gekko.options.COLDSTART = 1
            model.gekko.solve(disp=False, debug=0)

            die, allocation, dispersions = extract_solution(model, die, cells, threshold)
            assert die.netlist is not None, "No netlist associated to the die"  # Assertion to suppress Mypy error

            if plotting_options.joint_plot:
                vis_imgs[0].append(get_joint_floorplan_plot(die.netlist, allocation, die.bounding_box.shape))
            if plotting_options.separated_plot:
                vis_imgs[1].append(get_separated_floorplan_plot(die, allocation))
            print(i, end=" ", flush=True)

            if model.gekko.options.APPSTATUS == 1:
                print("\nThe solution was found.")
                break
            else:
                i += 1
        else:
            print(f"Maximum number of iterations ({max_iter}) reached! The solution was not found.")

    assert allocation is not None and dispersions is not None  # Assertion to supress Mypy error
    return die, allocation, dispersions, vis_imgs


def get_a(allocation: Allocation, module: Module, cell_index: int) -> float:
    """
    Get the allocation value of a module in a particular cell.
    If the module is part of the allocation of the cell, the value is returned.
    If not, but it has a single rectangle (this is useful for "fake" modules from non-fixed hard modules), then
    the allocation value is calculated.
    Else, 0.0 is returned
    :param allocation: the allocation
    :param module: the module
    :param cell_index: the index of the allocation rectangle
    :return: the area ratio of the module in the cell, as described
    """
    if module.name in allocation.allocations[cell_index].alloc:
        return allocation.allocations[cell_index].alloc[module.name]
    if isinstance(module, Module) and module.num_rectangles == 1:
        module_rect = module.rectangles[0]
        alloc_rect = allocation.allocations[cell_index].rect
        return alloc_rect.area_overlap(module_rect) / alloc_rect.area
    return 0.0


def optimize_allocation(die: Die, allocation: Allocation, dispersions: dict[str, float],
                        threshold: float, alpha: float, dispersion_function: DispersionFunction,
                        verbose: bool = False, plotting_options: PlottingOptions | None = None) \
        -> tuple[Die, Allocation, dict[str, float], tuple[list[Image.Image], list[Image.Image]]]:
    """
    Optimizes the given allocation to minimize the dispersion and the wire length of the floor plan
    :param die: die with netlist containing the modules with centroids initialized
    :param allocation: allocation to optimize
    :param dispersions: a dictionary which indicates the dispersion of each module
    :param threshold: hyperparameter between 0 and 1 to decide if allocations can be fixed
    :param alpha: hyperparameter between 0 and 1 to control the balance between dispersion and wire length.
    Smaller values will reduce the dispersion and increase the wire length, and greater ones the other way around
    :param dispersion_function: the function to use to calculate the dispersion of each module
    :param verbose: if True, the GEKKO optimization log is displayed
    :param plotting_options: plotting options
    :return: the optimal solution found:
    - die - die with netlist with the centroids of the modules updated
    - allocation - optimized allocation
    - dispersions - a dictionary which indicates the dispersion of each module
    - vis_imgs - if visualize is True, a list of images visualizing the optimization, otherwise, an empty list
    """
    assert die.netlist is not None, "No netlist associated to the die"

    n_cells = allocation.num_rectangles
    cells = [alloc.rect for alloc in allocation.allocations]

    bb = die.bounding_box.bounding_box

    model = Model()
    g = model.gekko  # Shortcut (reference)

    modules = []
    nonfixed_hard_modules = []
    for module in die.netlist.modules:
        m = module.name
        if not module.is_hard or module.is_fixed:
            modules.append(module)
        else:
            nonfixed_hard_modules.append(module)

            # "fake" modules
            for r, rectangle in enumerate(module.rectangles):
                mr = f"{m}_{r}"
                fake_module = Module(mr, hard=True, fixed=False)
                fake_module.add_rectangle(rectangle)
                fake_module.setup()
                fake_module.calculate_center_from_rectangles()
                modules.append(fake_module)
                dispersions[mr] = dispersions[m]

    # Centroid and dispersion of modules
    for module in modules:
        m = module.name
        assert module.center is not None
        if module.is_fixed:
            model.x[m] = module.center.x
            model.y[m] = module.center.y
        else:
            model.x[m] = g.Var(value=module.center.x, lb=bb.ll.x, ub=bb.ur.x, name=f"x_{m}")
            model.y[m] = g.Var(value=module.center.y, lb=bb.ll.y, ub=bb.ur.y, name=f"y_{m}")
            model.d[m] = g.Var(value=dispersions[m], lb=0, name=f"d_{m}")

    # Get neighbouring cells of all the cells
    neigh_cells: list[list[int]] = [[]] * n_cells
    for c in range(n_cells):
        neigh_cells[c] = get_neighbouring_cells(allocation, c)

    # Ratios of area of c used by module m
    for module in modules:
        m = module.name
        model.a[m] = {}
        for c in range(n_cells):
            a_mc = get_a(allocation, module, c)
            if module.is_fixed or \
                    a_mc > threshold and all(get_a(allocation, module, d) > threshold for d in neigh_cells[c]) or \
                    a_mc < 1 - threshold and all(get_a(allocation, module, d) < 1 - threshold for d in neigh_cells[c]):
                model.a[m][c] = a_mc
            else:
                model.a[m][c] = g.Var(value=a_mc, lb=0, ub=1, name=f"a_{m}_{c}")

    # Cell constraints
    for c in range(n_cells):
        # Cells cannot be over-occupied
        g.Equation(g.sum([model.a[m][c] for m in model.a.keys()]) <= 1)

    # Module constraints
    for module in modules:
        m = module.name

        # Modules must have sufficient area
        g.Equation(g.sum([cells[c].area * model.a[m][c] for c in range(n_cells)]) >= module.area())

        # Centroid of modules
        g.Equation(1 / module.area() * g.sum([cells[c].area * cells[c].center.x * model.a[m][c]
                                              for c in range(n_cells)]) == model.x[m])
        g.Equation(1 / module.area() * g.sum([cells[c].area * cells[c].center.y * model.a[m][c]
                                              for c in range(n_cells)]) == model.y[m])

        # Dispersion of soft modules
        if not module.is_hard:
            g.Equation(g.sum([cells[c].area * model.a[m][c] *
                              dispersion_function(model.x[m] - cells[c].center.x, model.y[m] - cells[c].center.y)
                              for c in range(n_cells)]) == model.d[m])

    for module in nonfixed_hard_modules:
        m = module.name
        assert module.center is not None
        model.x[m] = g.Var(value=module.center.x, lb=bb.ll.x, ub=bb.ur.x, name=f"x_{m}")
        model.y[m] = g.Var(value=module.center.y, lb=bb.ll.y, ub=bb.ur.y, name=f"y_{m}")

        model.a[m] = {}
        for c in range(n_cells):
            model.a[m][c] = g.Var(value=get_a(allocation, module, c), lb=0, ub=1, name=f"a_{m}_{c}")
            g.Equation(model.a[m][c] == g.sum([model.a[f"{m}_{r}"][c] for r in range(len(module.rectangles))]))

        for r, rectangle in enumerate(module.rectangles):
            mr = f"{m}_{r}"
            g.Equation(model.x[m] - model.x[mr] == module.center.x - rectangle.center.x)
            g.Equation(model.y[m] - model.y[mr] == module.center.y - rectangle.center.y)

            w, h = astuple(rectangle.shape)
            g.Equation(g.sum([cells[c].area * model.a[mr][c] *
                              (dispersion_function(h / w * (model.x[mr] - cells[c].center.x),
                                                   model.y[mr] - cells[c].center.y) if w < h
                               else dispersion_function(model.x[mr] - cells[c].center.x,
                                                        w / h * (model.y[mr] - cells[c].center.y)))
                              for c in range(n_cells)]) == model.d[mr])

    # Objective function: alpha * total wire length + (1 - alpha) * total dispersion

    # Total wire length
    for e in die.netlist.edges:
        if len(e.modules) == 2:  # Regular edges (we can avoid using extra variables)
            m0 = e.modules[0].name
            m1 = e.modules[1].name
            g.Minimize(alpha * e.weight * ((model.x[m0] - model.x[m1])**2 + (model.y[m0] - model.y[m1])**2) / 2)
        else:  # Hyperedges
            ex = g.Var(lb=0)
            g.Equation(g.sum([model.x[module.name] for module in e.modules]) / len(e.modules) == ex)
            ey = g.Var(lb=0)
            g.Equation(g.sum([model.y[module.name] for module in e.modules]) / len(e.modules) == ey)
            for module in e.modules:
                m = module.name
                g.Minimize(alpha * e.weight * ((ex - model.x[m])**2 + (ey - model.y[m])**2))

    # Total dispersion
    g.Minimize((1 - alpha) * g.sum(list(model.d.values())))

    die, allocation, dispersions, vis_imgs = solve_and_extract_solution(model, die, cells, threshold, verbose=verbose,
                                                                        plotting_options=plotting_options)
    return die, allocation, dispersions, vis_imgs


def glbfloor(die: Die, threshold: float, alpha: float,
             dispersion_function: DispersionFunction = lambda x, y: x**2 + y**2,
             max_iter: int | None = None, verbose: bool = False, plotting_options: PlottingOptions | None = None) \
        -> tuple[Die, Allocation]:
    """
    Calculates the initial allocation and optimizes it to minimize the dispersion and the wire length of the floor plan.
    Afterwards, the allocation is repeatedly refined and optimized until it cannot be further refined or the maximum
    number of iterations is reached
    :param die: die with netlist containing the modules with centroids initialized
    :param threshold: hyperparameter between 0 and 1 to decide if allocations must be refined
    :param alpha: hyperparameter between 0 and 1 to control the balance between dispersion and wire length.
    Smaller values will reduce the dispersion and increase the wire length, and greater ones the other way around
    :param dispersion_function: the function to use to calculate the dispersion of each module
    :param max_iter: maximum number of optimization iterations performed, or None to stop when no more refinements
    can be performed
    :param verbose: if True, the GEKKO optimization log and iteration numbers are displayed
    :param plotting_options: plotting options
    :return: the optimal solution found:
    - netlist - Netlist with the centroids of the modules updated.
    - allocation - Refined allocation with the ratio of each module in each cell of the grid.
    """
    assert die.netlist is not None, "No netlist associated to the die"

    joint_vis_imgs = []
    separated_vis_imgs = []
    durations = []

    allocation = create_initial_allocation(die)
    dispersions = calculate_dispersions(die.netlist.modules, allocation, dispersion_function)

    n_iter = 0

    if plotting_options is not None:
        if plotting_options.joint_plot:
            get_joint_floorplan_plot(die.netlist, allocation, die.bounding_box.shape). \
                save(f"{plotting_options.name}-joint-{n_iter}.gif")

        if plotting_options.separated_plot:
            get_separated_floorplan_plot(die, allocation, dispersions, alpha, draw_text=True). \
                save(f"{plotting_options.name}-separated-{n_iter}.png")

    n_iter += 1
    while max_iter is None or n_iter <= max_iter:
        if n_iter > 1:
            if allocation.must_be_refined(threshold):
                allocation = allocation.refine(threshold)
            else:
                break

        die, allocation, dispersions, vis_imgs = optimize_allocation(die, allocation, dispersions, threshold, alpha,
                                                                     dispersion_function, verbose, plotting_options)
        assert die.netlist is not None, "No netlist associated to the die"  # Assertion to suppress Mypy error

        if plotting_options is not None:
            if plotting_options.visualize:
                joint_vis_imgs.extend(vis_imgs[0])
                separated_vis_imgs.extend(vis_imgs[1])
                durations.extend([100] * (max(len(vis_imgs[0]), len(vis_imgs[1])) - 1) + [1000])

            if plotting_options.joint_plot:
                get_joint_floorplan_plot(die.netlist, allocation, die.bounding_box.shape). \
                    save(f"{plotting_options.name}-joint-{n_iter}.gif")

            if plotting_options.separated_plot:
                get_separated_floorplan_plot(die, allocation, dispersions, alpha, draw_text=True). \
                    save(f"{plotting_options.name}-separated-{n_iter}.png")

        if verbose:
            print(f"Iteration {n_iter} finished\n")

        n_iter += 1

    if plotting_options is not None and plotting_options.visualize:
        if plotting_options.joint_plot:
            joint_vis_imgs[0].save(f"{plotting_options.name}-joint-visualization.gif",
                                   save_all=True, append_images=joint_vis_imgs[1:], duration=durations)

        if plotting_options.separated_plot:
            separated_vis_imgs[0].save(f"{plotting_options.name}-separated-visualization.gif",
                                       save_all=True, append_images=separated_vis_imgs[1:], duration=durations)

    return die, allocation
