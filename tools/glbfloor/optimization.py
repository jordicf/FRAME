"""This file contains the code for optimizing the floor plans"""
from typing import Any, Callable

from PIL import Image
from gekko import GEKKO

from frame.geometry.geometry import Point, Rectangle
from frame.die.die import Die
from frame.allocation.allocation import AllocDescriptor, Alloc, Allocation, create_initial_allocation
from frame.netlist.module import Module

from tools.draw.draw import get_floorplan_plot as get_joint_floorplan_plot
from tools.glbfloor.plots import PlottingOptions, get_separated_floorplan_plot


class Model:
    """GEKKO model with variables"""
    gekko: GEKKO

    # Model variables
    # (without accurate type hints because GEKKO does not have type hints yet)
    x: Any
    y: Any
    d: Any
    a: Any

    def __init__(self):
        """Constructs the GEKKO object"""
        self.gekko = GEKKO(remote=False)


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


DispersionFunction = Callable[[float, float], float]


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
        assert module.center is not None
        dispersions[module.name] = 0.0
        for module_alloc in allocation.allocation_module(module.name):
            cell = allocation.allocation_rectangle(module_alloc.rect_index).rect
            area = cell.area * module_alloc.area_ratio
            dispersions[module.name] += area * dispersion_function(module.center.x - cell.center.x,
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
        for m, module in enumerate(die.netlist.modules):
            a_mc_val = get_value(model.a[m][c])
            if a_mc_val > 1 - threshold:
                alloc[module.name] = a_mc_val
        if alloc:  # add only if not empty
            allocation_list.append((cell, alloc, 0))
    allocation = Allocation(allocation_list)

    dispersions = {}
    for m, module in enumerate(die.netlist.modules):
        module.center = Point(get_value(model.x[m]), get_value(model.y[m]))
        dispersions[module.name] = get_value(model.d[m])

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

            if plotting_options.separated_plot:
                vis_imgs[0].append(get_joint_floorplan_plot(die.netlist, allocation, die.bounding_box.shape))
            if plotting_options.joint_plot:
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

    n_modules = die.netlist.num_modules
    module2m = {}
    for m, module in enumerate(die.netlist.modules):
        module2m[module.name] = m

    bb = die.bounding_box.bounding_box

    model = Model()
    g = model.gekko  # Shortcut (reference)

    # Centroid of modules
    model.x = g.Array(g.Var, n_modules, lb=bb.ll.x, ub=bb.ur.x)
    model.y = g.Array(g.Var, n_modules, lb=bb.ll.y, ub=bb.ur.y)

    # Dispersion of modules
    model.d = g.Array(g.Var, n_modules, lb=0)

    # Ratios of area of c used by module m
    model.a = g.Array(g.Var, (n_modules, n_cells), value=0, lb=0, ub=1)

    # Set initial values
    for m, module in enumerate(die.netlist.modules):
        assert module.center is not None
        model.x[m].value, model.y[m].value = module.center
        model.d[m].value = dispersions[module.name]

    for c, rect_alloc in enumerate(allocation.allocations):
        for module_name, area_ratio in rect_alloc.alloc.items():
            model.a[module2m[module_name]][c].value = area_ratio

    # Get neighbouring cells of all the cells
    neigh_cells: list[list[int]] = [[]] * n_cells
    for c in range(n_cells):
        neigh_cells[c] = get_neighbouring_cells(allocation, c)

    # Fix (make constant) cells that have an allocation close to one (or zero) and are completely surrounded by cells
    # that are also close to one (or zero). Modules with all the cells fixed, or with the fixed attribute set to True
    # are also fixed in the model.
    for m, module in enumerate(die.netlist.modules):
        const_module = True
        if not module.is_fixed:
            for c in range(n_cells):
                a_mc_val = get_value(model.a[m][c])
                if a_mc_val > threshold and all(get_value(model.a[m][d]) > threshold for d in neigh_cells[c]) or \
                   a_mc_val < 1 - threshold and all(get_value(model.a[m][d]) < 1 - threshold for d in neigh_cells[c]):
                    model.a[m][c] = a_mc_val
                elif const_module:
                    const_module = False
        if const_module:
            model.x[m] = get_value(model.x[m])
            model.y[m] = get_value(model.y[m])
            model.d[m] = get_value(model.d[m])
            for c in range(n_cells):
                model.a[m][c] = get_value(model.a[m][c])

    # Cell constraints
    for c in range(n_cells):
        # Cells cannot be over-occupied
        g.Equation(g.sum([model.a[m][c] for m in range(n_modules)]) <= 1)

    # Module constraints
    for m, module in enumerate(die.netlist.modules):
        # Modules must have sufficient area
        g.Equation(g.sum([cells[c].area * model.a[m][c] for c in range(n_cells)]) >= module.area())

        # Centroid of modules
        g.Equation(1 / module.area() * g.sum([cells[c].area * cells[c].center.x * model.a[m][c]
                                              for c in range(n_cells)]) == model.x[m])
        g.Equation(1 / module.area() * g.sum([cells[c].area * cells[c].center.y * model.a[m][c]
                                              for c in range(n_cells)]) == model.y[m])

        if not module.is_hard or module.is_fixed:
            # Dispersion of soft and fixed modules
            g.Equation(g.sum([cells[c].area * model.a[m][c] *
                              dispersion_function(model.x[m] - cells[c].center.x, model.y[m] - cells[c].center.y)
                              for c in range(n_cells)]) == model.d[m])
        else:  # Non-fixed hard modules
            raise NotImplementedError  # TODO: implement support for non-fixed hard modules

    # Objective function: alpha * total wire length + (1 - alpha) * total dispersion

    # Total wire length
    for e in die.netlist.edges:
        if len(e.modules) == 2:  # Regular edges (we can avoid using extra variables)
            m0 = module2m[e.modules[0].name]
            m1 = module2m[e.modules[1].name]
            g.Minimize(alpha * e.weight * ((model.x[m0] - model.x[m1])**2 + (model.y[m0] - model.y[m1])**2) / 2)
        else:  # Hyperedges
            ex = g.Var(lb=0)
            g.Equation(g.sum([model.x[module2m[module.name]] for module in e.modules]) / len(e.modules) == ex)
            ey = g.Var(lb=0)
            g.Equation(g.sum([model.y[module2m[module.name]] for module in e.modules]) / len(e.modules) == ey)
            for module in e.modules:
                m = module2m[module.name]
                g.Minimize(alpha * e.weight * ((ex - model.x[m])**2 + (ey - model.y[m])**2))

    # Total dispersion
    g.Minimize((1 - alpha) * g.sum([model.d[m] for m in range(die.netlist.num_modules)]))

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
            get_joint_floorplan_plot(die.netlist, allocation, die.bounding_box.shape).\
                save(f"{plotting_options.name}-joint-{n_iter}.gif")

        if plotting_options.separated_plot:
            get_separated_floorplan_plot(die, allocation, dispersions, alpha, draw_text=True).\
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
                get_joint_floorplan_plot(die.netlist, allocation, die.bounding_box.shape).\
                    save(f"{plotting_options.name}-joint-{n_iter}.gif")

            if plotting_options.separated_plot:
                get_separated_floorplan_plot(die, allocation, dispersions, alpha, draw_text=True).\
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
