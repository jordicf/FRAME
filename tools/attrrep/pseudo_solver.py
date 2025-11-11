import concurrent.futures
import numpy as np


def _possible_overlap(S1: float, S2: float, c1: float, c2: float, 
                     a1max: float = 2, a1min: float = 0.5,
                     a2max: float = 2, a2min: float = 0.5) -> bool:
    """
    Returns whether R1, a rectangle of area "S1" with lower-left vertex at (0, 0), 
    and R2, a rectangle with upper right vertex at (c1, c2) > (0, 0) of area "S2", 
    can overlap, for certain width and height assignments. Overlaps with no 
    area (e.g a line or a single corner) are not considered as overlaps.

    This is trivially true when there is no limit on the aspect ratio of
    the rectangles, as you can make one "very horizontal" and the other one
    "very vertical".

    The aspect ratio is defined as the height / width of the rectangles.

    Params:
        S1, S2: areas of R1 and R2
        c1, c2: (x, y) coordinates of the upper right vertex of R2
        a1max, a2max: maximum aspect ratio of R1, R2
        a1min, a2min: minimum aspect ratio of R1, R2
    """

    assert min(S1, S2, a1max, a1min, a2max, a2min) > 0, "All areas and aspect ratios must be positive"
    assert c1 >= 0, "R2 must be to the right of R1"

    if c2 < 0: c2 *= -1 # convert to symmetric case where c2 > 0
    
    # auxiliary functions
    f = lambda x: S1 / x # convex function
    g = lambda x: c2 - S2/(c1 - x) # concave
    h = lambda x: f(x) - g(x) # also convex

    xmax1 : float = np.sqrt(S1 / a1min) # maximum reachable x coordinate of R1
    xmin1 : float = np.sqrt(S1 / a1max) # minimum reachable x coordinate of right side of R1
    xmin2 : float = c1 - np.sqrt(S2 / a2min)
    xmax2 : float = c1 - np.sqrt(S2 / a2max)

    if c2 == 0: # simple case, rectangles are aligned
        return xmax1 > xmin2
    
    if c1 == 0:
        ymax1 : float = S1 / xmin1
        ymin2 : float = c2 + S2 / xmax2

        return ymax1 > ymin2
    

    if xmin1 > xmax2:
        return f(xmin1) > g(xmax2)
    elif xmax1 <= xmin2:
        return False
    else: # there is some overlap
        return h(max(xmin1, xmin2)) > 0 or h(min(xmax1, xmax2)) > 0

def _pairwise_overlap(h: np.ndarray, w: np.ndarray, 
                      centers: np.ndarray,
                      i: int, j: int) -> float:
    """
    Returns overlap between rectangles "i" and "j"

    Params:
        h, w: list of all rectangle heights and widths
        centers: list of (x,y) coordinates of all rectangle centers
        i, j: rectangle indices

    Returns:
        overlap between rectangles i & j
    """
    xi, yi, wi, hi = centers[i][0], centers[i][1], w[i], h[i]
    xj, yj, wj, hj = centers[j][0], centers[j][1], w[j], h[j]
    

    h_overlap: float = max(0, (wi + wj)/2 - abs(xj - xi))
    v_overlap: float = max(0, (hi + hj)/2 - abs(yj - yi))
    
    return h_overlap * v_overlap

def _pseudo_partial_derivative(h: np.ndarray, w: np.ndarray, 
                               centers: np.ndarray, i: int,
                               ar_max_i: float, ar_min_i: float,
                               fixed: set[int] = set()) -> float:
    """
    Given a layout and a rectangle 'i', it finds the width of rectangle
    'i' that minimizes the total overlap of the layout. The height
    of the rectangle is scaled to mantain a constant area.
    All the other rectangle widths (& heights) are not modified,
    so this computation is somewhat similar to a partial derivative.
    All rectangle centers remain unchanged.

    The found value is not guaranteed to be the optimum. It is found
    by sampling the range of possible w_i values, evaluating
    the overlap function on each sample, and then exploring
    the neighborhood of the best sample. This is repeated until
    there is no improvement.

    Params:
        h, w: list of heights and widths of every rectangle. Since areas
              are constant, given one variable you have the other.
        centers: list of (x,y) coordinates of the center of every rectangle
        i: index of the rectangle w.r.t the pseudo partial derivative is 
           calculated.
        ar_max_i, ar_min_i: aspect ratio bounds for the i-th rectangle

    Returns:
        best found value for the i-th width
    """

    if i in fixed:
        return w[i]

    # sampling resolution hyperparameter
    M: int = 5

    N: int = len(w)

    xi, yi, wi, hi = centers[i][0], centers[i][1], w[i], h[i]
    Ai: float = hi * wi

    # find all rectangles that can overlap with R_i if R_i is reshaped.
    overlap_candidates = set()

    for j in range(N):
        if j == i:
            continue

        xj, yj, wj, hj = centers[j][0], centers[j][1], w[j], h[j]
        Aj: float = hj * wj
        ar_j: float = hj / wj

        if _possible_overlap(Ai/4, Aj/4, abs(xi - xj), abs(yi - yj), ar_max_i,
                            ar_min_i, ar_j, ar_j):
            # if rectangle "i" can overlap with rectangle "j" for some
            # hi and wi, leaving hj, wj fixed, then add "j" to the set
            overlap_candidates.add(j)

    # define objective function
    # Naive idea: sum_{j in {1,...,N}\i} overlap(i,j)
    # Faster approach: sum_{j in overlap_candidates} overlap(i,j)
    def overlap(h: list[float], w: list[float], 
                centers: list[tuple[float, float]], i: int,
                overlap_candidates: set[int]):
        return sum(_pairwise_overlap(h, w, centers, i, j) for j in overlap_candidates)
    
    # minimize the objective function using a discrete sampling approach
    min_overlap: float = overlap(h, w, centers, i, overlap_candidates)
    best_w: float = wi

    wmin, wmax = np.sqrt(Ai / ar_max_i), np.sqrt(Ai / ar_min_i) # width bounds

    while True:
        width_lspace = np.linspace(wmin, wmax, M)
        overlaps = list[float]()
        
        for j in range(M):
            w[i] = width_lspace[j]
            h[i] = Ai / w[i]

            overlaps.append(overlap(h, w, centers, i, overlap_candidates))

        j_best: int = np.argmin(overlaps)

        if overlaps[j_best] < min_overlap:
            min_overlap = overlaps[j_best]
            best_w = width_lspace[j_best]
            
            wmin = width_lspace[max(0, j_best - 1)]
            wmax = width_lspace[min(M - 1, j_best + 1)]
        else:
            break

    # revert h_i, w_i variables to their original value
    w[i] = wi
    h[i] = hi
    
    return best_w

def _pseudo_gradient_serial(h: np.ndarray, w: np.ndarray, 
                            centers: np.ndarray, min_AR: list[float], 
                            max_AR: list[float], fixed: set[int] = {}):
    """
    Compute 'gradient' using a plain Python loop

    Params:
        h, w: list of heights, widths
        centers: list of (x, y) centers
        min_AR, max_AR: list with the max/min aspect ratio for 
                        each rectange
        fixed: set of fixed modules (they can't have
               their shapes modified)
    
    """
    N = len(w)
    return np.array([
        _pseudo_partial_derivative(
            h, w, centers, i,
            ar_max_i=max_AR[i], ar_min_i=min_AR[i], fixed=fixed
        ) for i in range(N)
    ])

def _parallel_worker(args):
    """Helper so executor.map can pickle arguments cleanly."""
    h, w, centers, i, ar_min_i, ar_max_i, fixed = args
    # pass copies of h, w; as they are modified in the function
    return _pseudo_partial_derivative(
        h.copy(), w.copy(), centers, i,
        ar_max_i=ar_max_i, ar_min_i=ar_min_i, fixed=fixed
    )

def _pseudo_gradient_parallel(
        h: np.ndarray, w: np.ndarray, centers: np.ndarray, 
        min_AR: np.ndarray, max_AR: np.ndarray, 
        fixed: set[int] = {}, max_workers=4):
    """
    Parallel version of _pseudo_gradient_serial(). Slower
    for small netlist sizes.
    
    Params:
        h, w: list of heights, widths
        centers: list of (x, y) centers
        min_AR, max_AR: list with the max/min aspect ratio for 
                        each rectange
        fixed: set of fixed modules (they can't have
               their shapes modified)
        max_workers: number of processes (defaults to os.cpu_count()).
    """
    N = len(w)
    tasks = [(h, w, centers, i, min_AR[i], max_AR[i], fixed) for i in range(N)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_parallel_worker, tasks))
    return np.array(results)

def solve_widths(centers: np.ndarray, h: np.ndarray, 
                 w: np.ndarray, areas: np.ndarray, min_AR: list[float], 
                 max_AR: list[float], fixed: set[int] = set() ,
                 hyperparams: dict = {},
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns widths and heights for each rectangle to minimize overlap, 
    given their areas, centers and aspect ratio bounds.

    Params:
        centers: list of tuples with the rectangle centers
        areas: list of areas
        min_AR: minimum aspect ratio (height/width) for each rectangle.
                This bound can be different for different rectangles.
        max_AR: maximum aspect ratio for each rectangle
        hyperparams: dictionary with hyperparameters
            -'epsilon': float: threshold to consider that a stationary point
                               has been reached
            -'lr': float: learning rate, in (0, 1] interval
            -'lr_decay': float: learning rate decay, in (0, 1] interval
            -'parallel': bool: use parallel version / not (default True)

    Returns:
        (h, w), where w is a list of widths and h a list of heights
    """

    hyperparams: dict = hyperparams.get('pseudo_solver', {})
    epsilon: float = hyperparams.get('epsilon', 1e-5) # sensitivity to change hyperparameter

    lr: float = hyperparams.get('lr', 1) # learning rate
    lr_decay: float = hyperparams.get('lr_decay', 0.95) # to avoid oscillation
    parallel: bool = hyperparams.get('parallel', True)
    pseudo_gradient = _pseudo_gradient_parallel if parallel else _pseudo_gradient_serial

    delta_w: np.ndarray = pseudo_gradient(h, w, centers, min_AR, 
                                          max_AR, fixed) - w
    
    w_new: np.ndarray = w + lr * delta_w

    similarity: float = abs(np.dot(w_new, w)) / \
            (np.linalg.norm(w_new) * np.linalg.norm(w)) # cosine distance
    
    w = w_new
    h = np.divide(areas, w)
    
    while similarity < 1 - epsilon:
        lr *= lr_decay
        delta_w = pseudo_gradient(h, w, centers, min_AR, 
                                  max_AR, fixed) - w
    
        w_new = w + lr * delta_w
        
        similarity = abs(np.dot(w_new, w)) / \
                    (np.linalg.norm(w_new) * np.linalg.norm(w))
        
        w = w_new
        h = np.divide(areas, w)

    return h, w
