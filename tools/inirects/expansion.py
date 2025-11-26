import networkx as nx
import numpy as np
from .repel_rectangles import repel_rectangles
from .pseudo_solver import solve_widths

# weighted center of mass aproach

def set_heights_widths(G: nx.Graph, h: np.ndarray, w: np.ndarray,
                       original_idx: list[int]) -> None:
    N: int = len(h)

    for i in range(N):
        G.nodes[original_idx[i]]['height'] = h[i]
        G.nodes[original_idx[i]]['width'] = w[i]

def set_centers(G: nx.Graph, centers: np.ndarray,
                 original_idx: list[int]) -> None:
    N: int = len(centers)

    for i in range(N):
        G.nodes[original_idx[i]]['center'] = centers[i].copy()

def pairwise_overlap(centers: np.ndarray,
                      h: np.ndarray, w: np.ndarray,
                      i: int, j: int) -> float:
    """
    Returns overlap between rectangles "i" and "j"
    """
    xi, yi, wi, hi = centers[i][0], centers[i][1], w[i], h[i]
    xj, yj, wj, hj = centers[j][0], centers[j][1], w[j], h[j]
    

    h_overlap: float = max(0, (wi + wj)/2 - abs(xj - xi))
    v_overlap: float = max(0, (hi + hj)/2 - abs(yj - yi))
    
    return h_overlap * v_overlap

def total_overlap(centers: np.ndarray,
                  h: np.ndarray, w: np.ndarray) -> float:
    """
    Calculates total overlap between all pairs of rectangles
    """
    
    N: int = len(w)
    accum_overlap: float = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            accum_overlap += pairwise_overlap(centers, h, w, i, j)

    return accum_overlap

def ar_bounds(centers: np.ndarray, areas: np.ndarray, 
               ar_min: list[float], ar_max: list[float], H: float, 
               W: float) -> tuple[list[float], list[float]]:
    """
    Finds the minimum and maximum aspect ratio each rectangle can
    have, given its center, so as not to exceed the die bound.
    The aspect ratio is also bounded by hard limits ar_max, ar_min.

    Params:
        centers: list of (x,y) coordinates of the rectangle centers
        areas: array of rectangle areas
        ar_min, ar_max: list with hard aspect ratio limits
        H, W: die dimensions

    Returns:
        min_AR, max_AR: arrays with the min and max possible aspect
                        ratio for each rectangle
    """
    N: int = len(centers)

    min_AR = list[float]()
    max_AR = list[float]()

    for i in range(N):
        x, y = centers[i]
        A: float = areas[i]
        
        w_bound: float = min(x, W - x) # largest possible half-width
        h_bound: float = min(y, H - y) # largest possible half-height

        if w_bound == 0 or h_bound == 0:
            min_AR.append(1)
            max_AR.append(1)
            continue
        
        ar_min_i: float = max(A / (4 * w_bound**2), ar_min[i]) # A / (4 * w_bound**2) is the aspect ratio at the max width
        ar_max_i: float = min((4 * h_bound**2) / A, ar_max[i]) # (4 * h_bound**2) / A is the aspect ratio at the max height
        min_AR.append(ar_min_i)
        max_AR.append(ar_max_i)
    
    min_AR = np.array(min_AR)
    max_AR = np.array(max_AR)

    return min_AR, max_AR

def expand(G: nx.Graph, centers: np.ndarray, heights: np.ndarray, 
           widths: np.ndarray, areas: np.ndarray, 
           ar_min: np.ndarray, ar_max: np.ndarray, H: float, W: float,
           fixed: set[int] = {}, hyperparams: dict = {}) -> None:
    
    hyperparams = hyperparams.get('expansion', {})

    # update epsilon
    hyperparams['epsilon'] = max(hyperparams.get('epsilon', 1e-5) * \
                                 hyperparams.get('epsilon_decay', 1),
                                 hyperparams.get('min_epsilon', 1e-5))
    
    # remove the terminals, as they have no area
    mask = np.ones(len(centers), dtype=bool)
    original_idx = []
    new_fixed = set[int]()

    for v in G.nodes:
        if G.nodes[v].get('terminal', False):
            mask[v] = False
        else:
            original_idx.append(v)
            if v in fixed:
                new_fixed.add(len(original_idx) - 1)

    # update function arguments
    centers = centers.copy()[mask]
    heights = heights.copy()[mask]
    widths = widths.copy()[mask]
    areas = areas.copy()[mask]
    ar_min = ar_min.copy()[mask]
    ar_max = ar_max.copy()[mask]
    fixed = new_fixed
        
    epsilon: float = hyperparams.get('epsilon', 1e-5) * H * W

    overlap: float = total_overlap(centers, heights, widths)

    while True:
        old_centers, old_h, old_w = centers, heights, widths
        prev_overlap = overlap

        centers: np.ndarray = repel_rectangles(centers, heights, 
                                               widths, H, W, fixed, 
                                               hyperparams)

        min_AR, max_AR = ar_bounds(centers, areas, ar_min, ar_max, H, W)

        heights, widths = solve_widths(centers, heights, widths, areas, min_AR,
                                       max_AR, fixed, hyperparams)

        overlap: float = total_overlap(centers, heights, widths)

        if overlap > prev_overlap - epsilon:
            break
    
    if overlap > prev_overlap:
        set_centers(G, old_centers, original_idx)
        set_heights_widths(G, old_h, old_w, original_idx)
    else:
        set_centers(G, centers, original_idx)
        set_heights_widths(G, heights, widths, original_idx)
