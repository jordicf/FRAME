from dataclasses import dataclass
from typing import Callable

from frame.allocation.allocation import Allocation
from frame.netlist.module import Module


@dataclass()
class DispersionFunction:
    # Function to calculate the dispersion in one axis, for example, lambda a, b: (a-b)**2.
    f1: Callable[[float, float], float]

    # Function to add the dispersions in the two axes, for example, lambda dx, dy: dx + dy.
    f2: Callable[[float, float], float]


def calculate_dispersions(modules: list[Module], allocation: Allocation, dispersion_function: DispersionFunction) \
        -> dict[str, tuple[float, float]]:
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
        dx, dy = 0.0, 0.0
        for module_alloc in allocation.allocation_module(module.name):
            cell = allocation.allocation_rectangle(module_alloc.rect_index).rect
            area = cell.area * module_alloc.area_ratio
            dx += area * dispersion_function.f1(module.center.x, cell.center.x)
            dy += area * dispersion_function.f1(module.center.y, cell.center.y)
        dispersions[module.name] = dx, dy
    return dispersions
