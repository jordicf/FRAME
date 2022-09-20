# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from typing import Union

from gekko import GEKKO
from gekko.gk_variable import GKVariable

from frame.die.die import Die
from frame.geometry.geometry import Point
from tools.draw.draw import get_floorplan_plot

GEKKOType = Union[float, GKVariable]


class Model:
    """GEKKO model with variables"""
    gekko: GEKKO

    # Model variables
    x: list[GEKKOType]
    y: list[GEKKOType]

    def __init__(self, die: Die):
        """Constructs the GEKKO object"""
        self.gekko = GEKKO(remote=False)
        g = self.gekko  # Shortcut (reference)

        assert die.netlist is not None, "No netlist associated to the die"

        die_bb = die.bounding_box.bounding_box

        self.x = [0.0] * die.netlist.num_modules
        self.y = [0.0] * die.netlist.num_modules

        for i, module in enumerate(die.netlist.modules):
            assert module.center is not None
            if module.is_fixed:
                self.x[i] = module.center.x
                self.y[i] = module.center.y
            else:
                self.x[i] = g.Var(value=module.center.x, lb=die_bb.ll.x, ub=die_bb.ur.x, name=f"x_{module.name}")
                self.y[i] = g.Var(value=module.center.y, lb=die_bb.ll.y, ub=die_bb.ur.y, name=f"y_{module.name}")


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
    if v is not float:
        try:
            v = float(v)
        except TypeError:
            raise ValueError(f"Could not get value of {v} (type: {type(v)}")
    return v


def extract_solution(model: Model, die: Die) -> Die:
    """
    Extracts the solution from the model
    :param model: the model
    :param die: the die with the netlist
    :return:
    - die - die with netlist with the centroids of the modules updated
    """
    assert die.netlist is not None, "No netlist associated to the die"

    for i, module in enumerate(die.netlist.modules):
        if not module.is_fixed:
            module.center = Point(get_value(model.x[i]), get_value(model.y[i]))
    return die


def solve_and_extract_solution(model: Model, die: Die, verbose: bool = False, visualize: str | None = None,
                               max_iter: int = 100) -> Die:
    """
    Solves the model's optimization problem, extracts the solution from it, and returns it
    :param model: the model
    :param die: the die with the netlist
    :param max_iter: maximum number of iterations for GEKKO
    :param verbose: if True, the GEKKO optimization log is displayed (not supported if visualize is True)
    :param visualize: if True, produce a GIF showing the optimization process
    :return:
    - die - die with netlist with the centroids of the modules updated
    """
    vis_imgs = []
    if visualize is None:
        model.gekko.options.MAX_ITER = max_iter
        model.gekko.solve(disp=verbose, debug=0)
        die = extract_solution(model, die)
    else:
        # See https://stackoverflow.com/a/73196238/10152624 for the method used here
        i = 0
        while i < max_iter:
            model.gekko.options.MAX_ITER = i
            model.gekko.options.COLDSTART = 1
            model.gekko.solve(disp=False, debug=0)

            die = extract_solution(model, die)
            assert die.netlist is not None, "No netlist associated to the die"  # Assertion to suppress Mypy error

            vis_imgs.append(get_floorplan_plot(die.netlist, die.bounding_box.shape))
            print(i, end=" ", flush=True)

            if model.gekko.options.APPSTATUS == 1:
                print("\nThe solution was found.")
                break
            else:
                i += 1
        else:
            print(f"Maximum number of iterations ({max_iter}) reached! The solution was not found.")

        vis_imgs[0].save(f"{visualize}.gif", save_all=True, append_images=vis_imgs[1:], duration=100)

    return die
