import itertools
import random
from .netlist import swapNetlist

def greedy(
    net: swapNetlist,
    verbose: bool = False,
) -> None:
    """Optimize the netlist using a greedy algorithm."""

    if verbose:
        print(f"Greedy: Initial HPWL {net.hpwl:.1f}")

    iter = 0  # Iteration counter
    pairs = list(itertools.combinations(net.movable, 2))
    nswaps = 1
    
    while nswaps > 0:
        iter += 1
        nswaps = 0
        smallest_delta = net.hpwl
        
        # Try all pairs in random order
        random.shuffle(pairs)
        for idx1, idx2 in pairs:
            delta_hpwl = net.swap_points(idx1, idx2)
            smallest_delta = min(smallest_delta, delta_hpwl)
            if delta_hpwl < 0:
                nswaps += 1
            else:
                net.swap_points(idx1, idx2)
                
        # Optional: print progress
        if verbose:
            print(f"Iter. {iter}, #swaps {nswaps}, HPWL: {net.hpwl:.1f}, min delta = {smallest_delta:.1f}")