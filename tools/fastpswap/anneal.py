# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).
"""Simulated annealing optimization for module centroid swapping."""

import math
import numpy as np
from typing import List
from numba import njit
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from .netlist import swapNetlist


@jitclass
class jitPoint:
    """A point in the netlist representing the position of a module."""

    x: float
    y: float
    nets: List[int]  # List of net IDs that this point belongs to

    def __init__(self, x: float, y: float, nets: List[int]) -> None:
        self.x = x
        self.y = y
        self.nets = NumbaList(nets)


@jitclass
class jitNet:
    """A net in the netlist."""

    weight: float  # Weight of the net
    points: List[int]  # List of point IDs that belong to this net
    hpwl: float  # Half-perimeter wire length (initialized to 0.0)
    prev_hpwl: float  # Previous HPWL before swapping

    def __init__(self, weight: float, points: List[int]) -> None:
        self.weight = weight
        self.points = NumbaList(points)
        self.hpwl = 0.0
        self.prev_hpwl = 0.0


@jitclass
class jitNetlist:
    _points: List[jitPoint]
    _nets: List[jitNet]
    _movable: List[int]  # List of movable point indices
    _hpwl: float  # Total HPWL of the netlist
    _prev_hpwl: float  # Previous total HPWL before swapping

    def __init__(
        self,
        points: List[jitPoint],
        nets: List[jitNet],
        movable: List[int],
    ) -> None:
        """Manual deep copy from swapNetlist to jitNetlist."""
        self._points = points
        self._nets = nets
        self._movable = movable
        hpwl = 0.0
        for n in nets:
            hpwl += self._compute_net_hpwl(n)
        self._hpwl = hpwl
        self._prev_hpwl = hpwl

    @property
    def movable(self) -> list[int]:
        """List of movable point indices."""
        return self._movable

    @property
    def points(self) -> list[jitPoint]:
        """List of points in the netlist."""
        return self._points

    @property
    def nets(self) -> list[jitNet]:
        """List of nets in the netlist."""
        return self._nets

    @property
    def hpwl(self) -> float:
        """Total half-perimeter wire length (HPWL) of the netlist."""
        return self._hpwl

    @hpwl.setter
    def hpwl(self, value: float) -> None:
        self._hpwl = value

    def _compute_net_hpwl(self, net: jitNet) -> float:
        """Compute the half-perimeter wire length (HPWL) of a net.
        It returns the computed HPWL."""
        net.prev_hpwl = net.hpwl
        xs = [self.points[p].x for p in net.points]
        ys = [self.points[p].y for p in net.points]
        net.hpwl = (max(xs) - min(xs) + max(ys) - min(ys)) * net.weight
        return net.hpwl

    def swap_points(self, idx1: int, idx2: int) -> float:
        """Swap two points and return the change in total HPWL."""
        self._prev_hpwl = self.hpwl
        p1, p2 = self.points[idx1], self.points[idx2]
        p1.x, p2.x = p2.x, p1.x
        p1.y, p2.y = p2.y, p1.y
        affected_nets = _merge_remove_common(p1.nets, p2.nets)
        delta_hpwl = 0.0
        for n in affected_nets:
            delta_hpwl -= self.nets[n].hpwl
            delta_hpwl += self._compute_net_hpwl(self.nets[n])
        self.hpwl += delta_hpwl
        return delta_hpwl

    def undo_swap(self, idx1: int, idx2: int) -> None:
        """Undo the swap of two points."""
        p1, p2 = self.points[idx1], self.points[idx2]
        p1.x, p2.x = p2.x, p1.x
        p1.y, p2.y = p2.y, p1.y
        affected_nets = _merge_remove_common(p1.nets, p2.nets)
        for n in affected_nets:
            self.nets[n].hpwl = self.nets[n].prev_hpwl
        self.hpwl = self._prev_hpwl


def simulated_annealing(
    net: swapNetlist,
    n_swaps: int,
    patience: int,
    target_acceptance: float = 0.5,
    temp_factor: float = 0.95,
    verbose: bool = False,
) -> None:
    if verbose:
        print("Creating JIT-compiled simulated annealing...")
    # Dummy definitions for Numba types
    points: NumbaList[jitPoint] = NumbaList([jitPoint(0.0, 0.0, [1])])
    nets: NumbaList[jitNet] = NumbaList([jitNet(1.0, [1])])

    # Real definitions
    points = NumbaList([jitPoint(p.x, p.y, p.nets) for p in net.points])
    nets: NumbaList[jitNet] = NumbaList([jitNet(n.weight, n.points) for n in net.nets])
    movable: NumbaList[int] = NumbaList(net.movable)
    jit_net = jitNetlist(points, nets, movable)

    # Fast simulated annealintg using Numba JIT compilation
    jit_simulated_annealing(
        jit_net, n_swaps, patience, target_acceptance, temp_factor, verbose
    )

    # Recover the optimized positions
    for i, p in enumerate(net.points):
        p.x = jit_net.points[i].x
        p.y = jit_net.points[i].y
    net.compute_total_hpwl()


@njit
def jit_simulated_annealing(
    net: jitNetlist,
    n_swaps: int,
    patience: int,
    target_acceptance: float,
    temp_factor: float,
    verbose: bool,
) -> None:
    """Optimize the netlist using simulated annealing.
    net is the swapNetlist to optimize,
    target_acceptance is the desired initial acceptance ratio (value in (0,0.95]),
    temp_factor is the factor by which the temperature decreases,
    patience is the number of iterations to perform without improvement,
    n_swaps is the number of swaps to perform per iteration and per movable point.
    Note that the total number of swaps per iteration is multiplied by the
    number of movable points."""

    if verbose:
        print("Running JIT-compiled simulated annealing...")
        print("Initial HPWL:", net.hpwl)

    n_swaps = n_swaps * len(net.movable)
    # Compute the initial temperature
    temp: float = _find_best_temperature(net, n_swaps, target_acceptance)

    # Initial solution
    best_xy = [(p.x, p.y) for p in net.points]
    best_hpwl = current_hpwl = net.hpwl

    if verbose:
        # print(f"Initially: Temperature {temp:.3f}, HPWL {best_hpwl:.1f}")
        print("Initially: Temperature", temp, "HPWL", best_hpwl)

    no_improvement = 0  # Number of iteration without improvement
    iter = 0  # Iteration counter
    best_avg = math.inf  # Conservative best average HPWL in one iteration
    while no_improvement < patience:
        iter += 1
        avg = 0.0
        # Perform n_swaps
        for _ in range(n_swaps):
            idx1, idx2 = _pick_two_randomly(net.movable)
            delta_hpwl = net.swap_points(idx1, idx2)

            if delta_hpwl < 0 or np.random.random() < np.exp(-delta_hpwl / temp):
                current_hpwl += delta_hpwl
                if current_hpwl < best_hpwl:
                    no_improvement = -1
                    best_hpwl = current_hpwl
                    best_xy = [(p.x, p.y) for p in net.points]
            else:
                # Swap back
                net.undo_swap(idx2, idx1)

            avg += current_hpwl
            
        avg /= n_swaps
        if avg >= best_avg:
            no_improvement += 1
        else:
            no_improvement = 0
            best_avg = avg
        if verbose:
            print(
                "Iter.",
                iter,
                "Temp.",
                temp,
                "HPWL: Avg",
                avg,
                "Best Avg",
                best_avg,
                "Best",
                best_hpwl,
            )
        temp = temp * temp_factor
        
    # Restore best solution
    for i, p in enumerate(net.points):
        p.x, p.y = best_xy[i]

    hpwl = 0.0
    for n in net.nets:
        hpwl += net._compute_net_hpwl(n)
    net.hpwl = hpwl


@njit
def _find_best_temperature(
    net: jitNetlist, nswaps: int, target_acceptance: float
) -> float:
    """Find the best temperature for simulated annealing.
    nswaps is the number of swaps performed to generate cost samples,
    target_acceptance is the desired acceptance ratio (value in (0,0.95])."""
    assert 0 < target_acceptance <= 0.95, "Target acceptance must be in (0, 0.95]"
    cost: list[float] = []  # incremental costs from the original location
    for _ in range(nswaps):
        idx1, idx2 = _pick_two_randomly(net.movable)
        cost.append(abs(net.swap_points(idx1, idx2)))
        net.undo_swap(idx2, idx1)  # Return to the original location

    # Compute target temperature
    nonzero_cost = [c for c in cost if c > 0]
    if not nonzero_cost:
        raise ValueError("No valid cost samples found")
    nonzero_cost.sort()
    idx = min(int(len(nonzero_cost) * target_acceptance), len(nonzero_cost) - 1)
    return -nonzero_cost[idx] / math.log(target_acceptance)


@njit
def _pick_two_randomly(choices: list[int]) -> tuple[int, int]:
    """Pick two different elements randomly from choices."""
    idx1 = idx2 = np.random.randint(0, len(choices))
    while idx2 == idx1:
        idx2 = np.random.randint(0, len(choices))
    return choices[idx1], choices[idx2]


@njit
def _merge_remove_common(list1: List[int], list2: List[int]) -> List[int]:
    """Merge two sorted lists into a single sorted list without duplicates."""
    merged: List[int] = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged.append(list1[i])
            i += 1
        elif list1[i] > list2[j]:
            merged.append(list2[j])
            j += 1
        else:
            i += 1
            j += 1
    while i < len(list1):
        merged.append(list1[i])
        i += 1
    while j < len(list2):
        merged.append(list2[j])
        j += 1
    return merged
