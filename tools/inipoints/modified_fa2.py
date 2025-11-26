import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
Function to give initial (illegal) floorplan using ForceAtlas2.

This code is taken from Networkx forceatlas2_layout implementation, 
and modified to support fixed vertices and canvas boundaries.

Original code can be found in
https://github.com/networkx/networkx/blob/main/networkx/drawing/layout.py
"""


def forceatlas2_layout(
    G: nx.Graph,
    W: float,
    H: float,
    pos: dict[int, np.ndarray] = {},
    scaling_factor: float | None = None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    linlog=False,
    seed=None,
    dim=2
):
    """
    Position nodes using a modified version of ForceAtlas2 force-directed
    layout algorithm, with support for fixed vertices.

    Params:
        G: netlist in graph format
    """

    if len(G) == 0:
        return {}
    
    # for bounding boxes
    def bbox(pos_dict):
        arr = np.array(list(pos_dict.values()))
        minv, maxv = arr.min(axis=0), arr.max(axis=0)
        return maxv - minv

    # internal function to compute layout
    def _fa2_core(G, pos_arr, fixed_mask):
        n = len(G)
        mass = np.zeros(n)
        size = np.zeros(n)
        adjust_sizes = node_size is not None
        nm = node_mass or {}
        ns = node_size or {}

        for idx, node in enumerate(G):
            mass[idx] = nm.get(node, G.degree(node) + 1)
            size[idx] = ns.get(node, 1)

        A = nx.to_numpy_array(G, weight=weight)
        speed, speed_efficiency = 1.0, 1.0
        swing, traction = 1.0, 1.0

        def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
            opt_jitter = 0.05 * np.sqrt(n)
            min_jitter = np.sqrt(opt_jitter)
            max_jitter = 10
            min_speed_efficiency = 0.05
            other = min(max_jitter, opt_jitter * traction / n**2)
            jitter = jitter_tolerance * max(min_jitter, other)
            if swing / traction > 2.0:
                if speed_efficiency > min_speed_efficiency:
                    speed_efficiency *= 0.5
                jitter = max(jitter, jitter_tolerance)
            if swing == 0:
                target_speed = np.inf
            else:
                target_speed = jitter * speed_efficiency * traction / swing
            if swing > jitter * traction:
                if speed_efficiency > min_speed_efficiency:
                    speed_efficiency *= 0.7
            elif speed < 1000:
                speed_efficiency *= 1.3
            max_rise = 0.5
            speed = speed + min(target_speed - speed, max_rise * speed)
            return speed, speed_efficiency

        for _ in range(max_iter):
            diff = pos_arr[:, None] - pos_arr[None]
            distance = np.linalg.norm(diff, axis=-1)

            if linlog:
                attraction = -np.log(1 + distance) / distance
                np.fill_diagonal(attraction, 0)
                attraction = np.einsum("ij, ij -> ij", attraction, A)
                attraction = np.einsum("ijk, ij -> ik", diff, attraction)
            else:
                attraction = -np.einsum("ijk, ij -> ik", diff, A)

            if distributed_action:
                attraction /= mass[:, None]

            tmp = mass[:, None] @ mass[None]
            if adjust_sizes:
                distance += -size[:, None] - size[None]
            d2 = np.maximum(0.001 * np.ones(len(distance)), distance**2)
            np.fill_diagonal(tmp, 0)
            np.fill_diagonal(d2, 1)
            factor = (tmp / d2) * scaling_ratio
            repulsion = np.einsum("ijk, ij -> ik", diff, factor)

            pos_centered = pos_arr - np.mean(pos_arr, axis=0)
            if strong_gravity:
                gravities = -gravity * mass[:, None] * pos_centered
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    unit_vec = pos_centered / np.linalg.norm(pos_centered, axis=-1)[:, None]
                unit_vec = np.nan_to_num(unit_vec, nan=0)
                gravities = -gravity * mass[:, None] * unit_vec

            update = attraction + repulsion + gravities
            swing += (mass * np.linalg.norm(pos_arr - update, axis=-1)).sum()
            traction += (0.5 * mass * np.linalg.norm(pos_arr + update, axis=-1)).sum()

            speed, speed_efficiency = estimate_factor(
                n, swing, traction, speed, speed_efficiency, jitter_tolerance
            )

            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = speed / (1 + np.sqrt(speed * swinging))
            factored_update = update * factor[:, None]

            # fixed nodes do not move
            factored_update[fixed_mask] = 0.0

            pos_arr += factored_update
            if abs(factored_update).sum() < 1e-10:
                break

        return pos_arr

    # initial position
    np.random.seed(seed)
    pos_arr = np.array([pos.get(node, np.random.rand(dim) * np.array([W, H])) for node in G]) # random positions for unassigned vertices
    nodes = list(G)

    if scaling_factor is None:
        # run with no fixed vertices to estimate scale
        tmp_pos_arr = _fa2_core(G, pos_arr.copy(), np.zeros(len(G), dtype=bool))
        free_box = bbox(dict(zip(nodes, tmp_pos_arr)))


        # scaling factor
        fixed_box = (W, H)
        ratios = []
        for fb, fx in zip(free_box, fixed_box):
            if fx > 0:
                ratios.append(fb / fx)
        scaling_factor = min(ratios) if ratios else 1.0

    # scale initial positions
    pos_arr *= scaling_factor

    # save fix coordinates to restore
    fixed_pos = {mod: center for mod, center in pos.items()}

    # second run with fixed vertices
    fixed_mask = np.array([G.nodes[node].get('fixed', False) for node in nodes])
    pos_arr = _fa2_core(G, pos_arr, fixed_mask)

    # undo scaling and restore fixed positions
    pos_arr /= scaling_factor
    pos = dict(zip(nodes, pos_arr))
    for node, p in fixed_pos.items():
        pos[node] = np.array(p)  # restore exactly (avoid numerical imprecisions)

    # clamp non-fixed vertices within (0, W) x (0, H)
    if W is not None and H is not None:
        out_nodes = []
        for node in nodes:
            if not G.nodes[node].get('fixed', False):
                x, y = pos[node]
                h = G.nodes[node].get("height", 0)
                w = G.nodes[node].get("width", 0)

                if h is None or w is None:
                    h = w = np.sqrt(G.nodes[node].get("area", 0))

                newx, newy = max(w/2, min(W - w/2, x)), max(h/2, min(H - h/2, y))

                if newx != x or newy != y:
                    out_nodes.append(node)

                pos[node] = np.array([newx, newy])
        
        # if out_nodes:
        #     print(f"Nodes outside range clamped: {out_nodes}")

    pos = {v: list(map(float, p)) for v, p in pos.items()}
    return pos, scaling_factor  
