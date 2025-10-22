import math
import numpy as np
from .netlist import swapNetlist


def simulated_annealing(
    net: swapNetlist,
    n_swaps: int,
    patience: int,
    target_acceptance: float = 0.5,
    temp_factor: float = 0.95,
    verbose: bool = False,
) -> None:
    """Optimize the netlist using simulated annealing.
    net is the swapNetlist to optimize,
    target_acceptance is the desired initial acceptance ratio (value in (0,0.95]),
    temp_factor is the factor by which the temperature decreases,
    patience is the number of iterations to perform without improvement,
    n_swaps is the number of swaps to perform per iteration and per movable point.
    Note that the total number of swaps per iteration is multiplied by the
    number of movable points."""

    n_swaps = n_swaps * len(net.movable)

    # Compute the initial temperature
    temp = _find_best_temperature(net, n_swaps, target_acceptance)

    # Initial solution
    best_xy = [(p.x, p.y) for p in net.points]
    best_hpwl = current_hpwl = net.hpwl

    if verbose:
        print(f"Initially: Temperature {temp:.3f}, HPWL {best_hpwl:.5f}")

    no_improvement = 0  # Number of iteration without improvement
    iter = 0  # Iteration counter
    best_avg = math.inf  # Conservative best average HPWL in one iteration
    while no_improvement < patience:
        iter += 1
        avg = 0.0
        # Perform n_swaps swaps
        for _ in range(n_swaps):
            idx1, idx2 = np.random.choice(net.movable, 2, replace=False)
            delta_hpwl = net.swap_points(idx1, idx2)
            new_hpwl = current_hpwl + delta_hpwl
            avg += new_hpwl
            if delta_hpwl < 0 or np.random.random() < np.exp(-delta_hpwl / temp):
                current_hpwl = new_hpwl
                if new_hpwl < best_hpwl:
                    no_improvement = -1
                    best_hpwl = new_hpwl
                    best_xy = [(p.x, p.y) for p in net.points]
            else:
                # Swap back
                net.swap_points(idx1, idx2)
        avg /= n_swaps
        if avg >= best_avg:
            no_improvement += 1
        else:
            no_improvement = 0
            best_avg = avg
        # Optional: print progress
        if verbose:
            print(
                f"Iter. {iter}, Temp. {temp:.3f}, HPWL: Avg {avg:.1f}, Best Avg {best_avg:.1f}, Best {best_hpwl:.1f}"
            )
        temp = temp * temp_factor
    # Restore best solution
    for i, p in enumerate(net.points):
        p.x, p.y = best_xy[i]
    net.hpwl = sum(net.compute_net_hpwl(n) for n in net.nets)


def _find_best_temperature(
    net: swapNetlist, nswaps: int, target_acceptance: float
) -> float:
    """Find the best temperature for simulated annealing.
    nswaps is the number of swaps performed to generate cost samples,
    target_acceptance is the desired acceptance ratio (value in (0,0.95])."""
    assert 0 < target_acceptance <= 0.95, "Target acceptance must be in (0, 0.95]"
    cost: list[float] = []  # incremental costs from the original location
    for _ in range(nswaps):
        idx1, idx2 = np.random.choice(net.movable, 2, replace=False)
        cost.append(abs(net.swap_points(idx1, idx2)))
        net.swap_points(idx1, idx2)  # Return to the original location

    # Compute target temperature
    cost.sort()
    idx = min(int(len(cost) * target_acceptance), len(cost) - 1)
    return -cost[idx] / math.log(target_acceptance)
