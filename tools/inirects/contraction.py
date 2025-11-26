import numpy as np
import networkx as nx

def update_hyperedge_centroids(G: nx.Graph) -> None:
    """
    Moves all dummy vertices (used to represent hyperedges, which are 
    multiple module nets) to the center of mass of their neighbors.

    Params:
        G: netlist in graph format
    """       

    for u in G.nodes:
        if not G.nodes[u].get("dummy", False):
            continue
        
        neighbors = list[int](G.neighbors(u))
        
        centers = list[np.array]()

        for v in neighbors:
            center: np.array = G.nodes[v].get('center')
            centers.append(center)

        centroid: np.array = np.mean(centers, axis=0)

        G.nodes[u]['center'] = centroid

def contract(G: nx.Graph, H: float, W: float, fixed: set[int] = set(),
             hyperparams: dict = {}) -> None:

    """
    Calculates attraction force for each rectangle. Rectangle shapes
    and areas are not used in this function.

    Params:
        G: netlist in graph format
        fixed: set of fixed modules

        hyperparams: dictionary with necessary hyperparameters. Valid
                     hyperparameters are
            - step_length (float): relative step length, in [0, 1] range
            - step_length_decay (float): constant to multiply "step_length"
                                         by at the end of the function call
            - step_length_max (float): max value of the step length (default 1)
            - step_length_min (float): min value of the step length (default 0)

    Returns: None

    NOTE: although it can be parallelized, it is < 1% of total runtime, 
          so it is pointless
    """

    # get specific hyperparameters for this function
    hyperparams: dict = hyperparams.get('contraction', {})

    alpha: float = hyperparams.get('step_length', 0.5) # the higher, the stronger the contraction
    niter: int = max(1, hyperparams.get('niter', 1))

    for _ in range(niter):
        # move dummy vertices to their correct place
        update_hyperedge_centroids(G)

        # calculate target point for each center
        for v in G.nodes:
            if G.nodes[v].get("dummy", False) or v in fixed:
                continue
            
            center_v: np.ndarray = G.nodes[v]['center']
            
            f = np.array([0.0, 0.0]) #force

            neighbors = list[int](G.neighbors(v))
            for neig in neighbors:
                d_pos: np.ndarray = G.nodes[neig].get('center') - center_v
                f += G[v][neig].get('weight') * d_pos / max(np.linalg.norm(d_pos), 1e-3)

            G.nodes[v]['target_point'] = f + center_v

        for v in G.nodes:
            if G.nodes[v].get("dummy", False) or v in fixed:
                continue

            targ_pt = G.nodes[v]['target_point']
            new_center = G.nodes[v]['center'] + alpha * (targ_pt - G.nodes[v]['center'])

            # make sure new position is legal
            h = G.nodes[v]['height']
            w = G.nodes[v]['width']

            new_center[0] = max(w/2, min(W - w/2, new_center[0]))
            new_center[1] = max(h/2, min(H - h/2, new_center[1]))

            G.nodes[v]['center'] = new_center

    if "niter" in hyperparams and "niter_decay" in hyperparams:
        hyperparams["niter"] -= min(niter - 1, hyperparams["niter_decay"])
    
    if "step_length" in hyperparams and "step_length_decay" in hyperparams:
        hyperparams["step_length"] *= hyperparams["step_length_decay"]


    hyperparams['step_length'] = alpha * hyperparams.get('step_length_decay', 1)
    
    hyperparams['step_length'] = max(hyperparams.get('step_length_min', 0),
                                     hyperparams['step_length'])
