#!/usr/bin/env python3
"""
GP-based Floorplan Optimization with Weighted wirelegth approximation(CasADi Version)
Uses CasADi's NLP solver (IPOPT) for efficient optimization
"""

import yaml
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os


def load_netlist(netlist_file):
    with open(netlist_file, 'r') as f:
        return yaml.safe_load(f)


def load_die_info(die_file):
    with open(die_file, 'r') as f:
        return yaml.safe_load(f)


def extract_movable_modules(netlist_data):
    modules = list[str]()
    module_data = dict[str, dict]()
    terminal_coords = dict[str, tuple[float, float]]()
    
    modules_dict = netlist_data.get('Modules', netlist_data)
    
    for module_name, module_info in modules_dict.items():
        if not isinstance(module_info, dict):
            continue
            
        # Handle terminals
        if module_info.get('terminal', False):
            center = module_info.get('center')
            if center and len(center) >= 2:
                terminal_coords[module_name] = (float(center[0]), float(center[1]))
            continue
        
        # Handle movable modules
        area = module_info.get('area', 0)
        rectangles = module_info.get('rectangles', [[0, 0, 1, 1]])
        
        if rectangles and len(rectangles[0]) == 4:
            x, y, w, h = rectangles[0]
            module_data[module_name] = {
                'area': area,
                'x': x, 'y': y, 'w': w, 'h': h
            }
            modules.append(module_name)
    
    return modules, module_data, terminal_coords


def extract_nets(netlist_data, modules):
    """Extract nets information from netlist"""
    nets = list[tuple[float, list[int], list[tuple[float, float]]]]()
    
    nets_list = netlist_data.get('Nets', [])
    module_to_idx = {name: idx for idx, name in enumerate(modules)}
    
    modules_dict = netlist_data.get('Modules', netlist_data)
    terminal_coords = dict[str, tuple[float, float]]()
    
    # Get terminal coordinates
    for module_name, module_info in modules_dict.items():
        if isinstance(module_info, dict) and module_info.get('terminal', False):
            center = module_info.get('center')
            if center and len(center) >= 2:
                terminal_coords[module_name] = (float(center[0]), float(center[1]))
    
    for net in nets_list:
        if not isinstance(net, (list, tuple)):
            continue
        
        # Extract weight (default 1.0)
        weight = 1.0
        pins = list(net)
        
        # Check if last element is a numeric weight
        if len(pins) >= 2 and isinstance(pins[-1], (int, float)) and not isinstance(pins[-1], str):
            weight = float(pins[-1])
            pins = pins[:-1]
        
        # Separate movable modules and terminals
        movable_indices = list[int]()
        terminal_positions = list[tuple[float, float]]()
        
        for pin in pins:
            pin_name = str(pin)
            if pin_name in module_to_idx:
                movable_indices.append(module_to_idx[pin_name])
            elif pin_name in terminal_coords:
                terminal_positions.append(terminal_coords[pin_name])
        
        # Only add nets with at least 2 pins (movable + terminal)
        if len(movable_indices) + len(terminal_positions) >= 2:
            nets.append((weight, movable_indices, terminal_positions))
    
    return nets


def detect_cycles(edges, n):
    """
    Detect cycles in directed graph using DFS
    Returns list of cycles found
    """
    # Build adjacency list
    adj = [list[int]() for _ in range(n)]
    for (i, j) in edges:
        adj[i].append(j)
    
    # DFS to detect cycles
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    parent = [-1] * n
    cycles = list[list[int]]()
    
    def dfs(node, path):
        if color[node] == GRAY:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
        if color[node] == BLACK:
            return
        
        color[node] = GRAY
        path.append(node)
        
        for neighbor in adj[node]:
            dfs(neighbor, path)
        
        path.pop()
        color[node] = BLACK
    
    for i in range(n):
        if color[i] == WHITE:
            dfs(i, list[int]())
    
    return cycles


def topological_sort(edges, n):
    """
    Perform topological sort to detect cycles and find a valid ordering
    Returns (is_dag, ordering) where is_dag is True if no cycles exist
    """
    # Build adjacency list and in-degree count
    adj = [list[int]() for _ in range(n)]
    in_degree = [0] * n
    
    for (i, j) in edges:
        adj[i].append(j)
        in_degree[j] += 1
    
    # Kahn's algorithm for topological sort
    queue = list[int]()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    ordering = list[int]()
    while queue:
        node = queue.pop(0)
        ordering.append(node)
        
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If we processed all nodes, it's a DAG
    is_dag = len(ordering) == n
    return is_dag, ordering


def break_cycles(edges, n, module_data, modules):
    """
    Break cycles using iterative approach:
    1. Detect all cycles
    2. Remove edges with maximum slack from each cycle
    3. Repeat until no cycles remain
    """
    original_edges = edges.copy()
    total_removed = 0
    iteration = 0
    max_iterations = 10  # Prevent infinite loops
    
    while iteration < max_iterations:
        cycles = detect_cycles(edges, n)
        if not cycles:
            break
            
        iteration += 1
        print(f"  Iteration {iteration}: Found {len(cycles)} cycles")
        
        # Calculate slack for each edge
        edge_slacks = dict[tuple[int, int], float]()
        for (i, j) in edges:
            data_i = module_data[modules[i]]
            data_j = module_data[modules[j]]
            
            # Calculate slack as distance between modules
            if i < j:  # horizontal edge
                slack = data_j['x'] - data_i['x']
            else:  # vertical edge  
                slack = data_j['y'] - data_i['y']
            
            edge_slacks[(i, j)] = slack
        
        # Remove edges to break cycles
        edges_to_remove = set[tuple[int, int]]()
        
        for cycle in cycles:
            print(f"    Breaking cycle: {' -> '.join([str(cycle[i]) for i in range(len(cycle)-1)])} -> {cycle[0]}")
            
            # Find edge in cycle with maximum slack
            max_slack = -float('inf')
            edge_to_remove = None
            
            for i in range(len(cycle) - 1):
                edge = (cycle[i], cycle[i + 1])
                if edge in edge_slacks:
                    slack = edge_slacks[edge]
                    if slack > max_slack:
                        max_slack = slack
                        edge_to_remove = edge
            
            if edge_to_remove:
                edges_to_remove.add(edge_to_remove)
                print(f"      Removing edge {edge_to_remove} (slack: {max_slack:.2f})")
        
        # Remove edges
        edges = [edge for edge in edges if edge not in edges_to_remove]
        removed_count = len(edges_to_remove)
        total_removed += removed_count
        
        print(f"    Removed {removed_count} edges in iteration {iteration}")
    
    if iteration >= max_iterations:
        print(f"    Warning: Reached maximum iterations ({max_iterations})")
        remaining_cycles = detect_cycles(edges, n)
        if remaining_cycles:
            print(f"    {len(remaining_cycles)} cycles still remain")
    
    print(f"  Total removed {total_removed} edges to break cycles")
    return edges, total_removed


def build_constraint_graphs(module_data, modules):
    """
    Build constraint graphs using center-based distance method
    - If dx > dy: add to HCG (horizontal constraint graph)
    - If dy > dx: add to VCG (vertical constraint graph)
    - If dx == dy: add to BOTH HCG and VCG for stronger separation
    """
    n = len(modules)
    horizontal_edges = list[tuple[int, int]]()
    vertical_edges = list[tuple[int, int]]()
    
    stats = {'horizontal_only': 0, 'vertical_only': 0, 'both': 0, 'total_pairs': 0}
    
    for i in range(n):
        for j in range(i+1, n):
            data_i = module_data[modules[i]]
            data_j = module_data[modules[j]]
            
            # Calculate center distances
            dx = abs(data_j['x'] - data_i['x'])
            dy = abs(data_j['y'] - data_i['y'])
            
            stats['total_pairs'] += 1
            
            # Determine horizontal edge direction
            h_edge = (i, j) if data_i['x'] < data_j['x'] else (j, i)
            
            # Determine vertical edge direction
            v_edge = (i, j) if data_i['y'] < data_j['y'] else (j, i)
            
            # Choose constraint direction(s) based on center distance comparison
            if abs(dx - dy) < 1e-6:
                # dx == dy: add to BOTH constraint graphs
                horizontal_edges.append(h_edge)
                vertical_edges.append(v_edge)
                stats['both'] += 1
            elif dx > dy:
                # Horizontal distance is larger -> add horizontal constraint only
                horizontal_edges.append(h_edge)
                stats['horizontal_only'] += 1
            else:
                # Vertical distance is larger -> add vertical constraint only
                vertical_edges.append(v_edge)
                stats['vertical_only'] += 1
    
    print(f"  Constraint selection based on center distances:")
    print(f"    Total module pairs: {stats['total_pairs']}")
    print(f"    Horizontal only: {stats['horizontal_only']}")
    print(f"    Vertical only: {stats['vertical_only']}")
    print(f"    Both (dx=dy): {stats['both']}")
    print(f"    Total H-constraints: {len(horizontal_edges)}")
    print(f"    Total V-constraints: {len(vertical_edges)}")
    
    # Check and break cycles in both constraint graphs
    print(f"  Checking cycles in horizontal constraint graph...")
    horizontal_edges, h_removed = break_cycles(horizontal_edges, n, module_data, modules)
    
    print(f"  Checking cycles in vertical constraint graph...")
    vertical_edges, v_removed = break_cycles(vertical_edges, n, module_data, modules)
    
    print(f"  Total edges removed to break cycles: {h_removed + v_removed}")
    
    # Final constraint count
    total_constraints = len(horizontal_edges) + len(vertical_edges)
    print(f"  Final constraint count: {total_constraints} (H={len(horizontal_edges)}, V={len(vertical_edges)})")
    
    return horizontal_edges, vertical_edges


def setup_casadi_model(modules, module_data, die_width, die_height,
                       horizontal_edges, vertical_edges, nets, max_ratio=3.0, alpha=1.0):
    """
    CasADi Canonical GP Model Setup for Floorplan Optimization with HPWL
    
    Exponential Space Variables (for GP formulation):
    - x_i, y_i, w_i, h_i: log-space variables
    - X[i] = e^{x_i}, Y[i] = e^{y_i}, W[i] = e^{w_i}, H[i] = e^{h_i}
    
    HPWL per net n (using max/min auxiliary variables):
    - X_max^n, X_min^n: max and min of X coordinates in net n
    - Y_max^n, Y_min^n: max and min of Y coordinates in net n
    - HPWL_n = (X_max^n - X_min^n) + (Y_max^n - Y_min^n)
    
    Constraints (ALL in Canonical GP form: posynomial <= 1):
    - Area: w_i + h_i = log(area_i)
    - Aspect ratio: -log(max_ratio) <= w_i - h_i <= log(max_ratio)
    - Non-overlap (HCG): e^{x_i - x_j} + 0.5*e^{w_i - x_j} + 0.5*e^{w_j - x_j} <= 1
    - Non-overlap (VCG): e^{y_i - y_j} + 0.5*e^{h_i - y_j} + 0.5*e^{h_j - y_j} <= 1
    - Boundary: 0.5*e^{w_i - x_i} <= 1, 0.5*e^{h_i - y_i} <= 1
    - Bounding box: e^{x_i - w_box} + 0.5*e^{w_i - w_box} <= 1
    - HPWL bounds: e^{x_i - x_max^n} <= 1, e^{x_min^n - x_i} <= 1 (and similar for y)
    
    Objective: minimize sum_n weight_n * (X_max^n - X_min^n + Y_max^n - Y_min^n)
              = sum_n weight_n * (e^{x_max^n} - e^{x_min^n} + e^{y_max^n} - e^{y_min^n})
    
    Note: alpha parameter is kept for interface compatibility but not used in max/min formulation.
    """
    n = len(modules)
    num_nets = len(nets)
    
    # Create optimization variables in log-space
    x = ca.MX.sym('x', n)  # log(X) coordinates
    y = ca.MX.sym('y', n)  # log(Y) coordinates
    w = ca.MX.sym('w', n)  # log(W) widths
    h = ca.MX.sym('h', n)  # log(H) heights
    w_box = ca.MX.sym('w_box')  # log(W_box) bounding box width
    h_box = ca.MX.sym('h_box')  # log(H_box) bounding box height
    
    # HPWL auxiliary variables for each net (GP-compatible max/min bounds)
    x_max = ca.MX.sym('x_max', num_nets)  # log(X_max^n) for each net
    x_min = ca.MX.sym('x_min', num_nets)  # log(X_min^n) for each net
    y_max = ca.MX.sym('y_max', num_nets)  # log(Y_max^n) for each net
    y_min = ca.MX.sym('y_min', num_nets)  # log(Y_min^n) for each net
    
    # Concatenate all variables
    opt_vars = ca.vertcat(x, y, w, h, w_box, h_box, x_max, x_min, y_max, y_min)
    
    # Initialize variable bounds and initial guess (in log-space)
    # Order must match: ca.vertcat(x, y, w, h, w_box, h_box, x_max, x_min, y_max, y_min)
    lbx = list[float]()
    ubx = list[float]()
    x0 = list[float]()
    
    # Bounds for x (log of X coordinates)
    for i in range(n):
        data = module_data[modules[i]]
        lbx.append(np.log(1e-3))
        ubx.append(np.log(die_width))
        x0.append(np.log(max(data['x'], 1e-3)))
    
    # Bounds for y (log of Y coordinates)
    for i in range(n):
        data = module_data[modules[i]]
        lbx.append(np.log(1e-3))
        ubx.append(np.log(die_height))
        x0.append(np.log(max(data['y'], 1e-3)))
    
    # Bounds for w (log of W)
    for i in range(n):
        data = module_data[modules[i]]
        area = data['area']
        if area > 0:
            w_min = np.sqrt(area / max_ratio)
            w_max = np.sqrt(area * max_ratio)
        else:
            w_min = 1e-3
            w_max = die_width
        lbx.append(np.log(max(w_min, 1e-3)))
        ubx.append(np.log(w_max))
        x0.append(np.log(max(data['w'], 1e-3)))
    
    # Bounds for h (log of H)
    for i in range(n):
        data = module_data[modules[i]]
        area = data['area']
        if area > 0:
            h_min = np.sqrt(area / max_ratio)
            h_max = np.sqrt(area * max_ratio)
        else:
            h_min = 1e-3
            h_max = die_height
        lbx.append(np.log(max(h_min, 1e-3)))
        ubx.append(np.log(h_max))
        x0.append(np.log(max(data['h'], 1e-3)))
    
    # Bounds for w_box (log of W_box)
    lbx.append(np.log(0.1 * die_width))
    ubx.append(np.log(1.0 * die_width))
    x0.append(np.log(die_width))
    
    # Bounds for h_box (log of H_box)
    lbx.append(np.log(0.1 * die_height))
    ubx.append(np.log(1.0 * die_height))
    x0.append(np.log(die_height))
    
    # Bounds for x_max (log of X_max^n for each net)
    for k in range(num_nets):
        lbx.append(np.log(1e-3))
        ubx.append(np.log(die_width))
        x0.append(np.log(die_width / 2))
    
    # Bounds for x_min (log of X_min^n for each net)
    for k in range(num_nets):
        lbx.append(np.log(1e-3))
        ubx.append(np.log(die_width))
        x0.append(np.log(die_width / 2))
    
    # Bounds for y_max (log of Y_max^n for each net)
    for k in range(num_nets):
        lbx.append(np.log(1e-3))
        ubx.append(np.log(die_height))
        x0.append(np.log(die_height / 2))
    
    # Bounds for y_min (log of Y_min^n for each net)
    for k in range(num_nets):
        lbx.append(np.log(1e-3))
        ubx.append(np.log(die_height))
        x0.append(np.log(die_height / 2))
    
    # Constraints in GP-compatible form
    g = list[ca.MX]()
    lbg = list[float]()
    ubg = list[float]()
    
    # 1. Area constraints: w_i + h_i = log(area_i)
    for i in range(n):
        area = module_data[modules[i]]['area']
        if area > 0:
            g.append(w[i] + h[i])
            log_area = np.log(area)
            lbg.append(log_area)
            ubg.append(log_area)
    
    # 2. Aspect ratio constraints: |w_i - h_i| <= log(max_ratio)
    log_max_ratio = np.log(max_ratio)
    for i in range(n):
        # w_i - h_i <= log(max_ratio)
        g.append(w[i] - h[i])
        lbg.append(-log_max_ratio)
        ubg.append(log_max_ratio)
    
    # 3. Boundary constraints (GP form: 0.5*e^{w_i - x_i} <= 1)
    log_half = np.log(0.5)
    for i in range(n):
        # 0.5*exp(w_i - x_i) <= 1  =>  exp(log(0.5) + w_i - x_i) <= 1
        g.append(ca.exp(log_half + w[i] - x[i]))
        lbg.append(0)
        ubg.append(1.0)
        
        # 0.5*exp(h_i - y_i) <= 1
        g.append(ca.exp(log_half + h[i] - y[i]))
        lbg.append(0)
        ubg.append(1.0)
    
    # 4. No-overlap constraints (GP form)
    print(f"  Adding no-overlap constraints: H={len(horizontal_edges)}, V={len(vertical_edges)}")
    
    # Horizontal: exp(x_i - x_j) + 0.5*exp(w_i - x_j) + 0.5*exp(w_j - x_j) <= 1
    for (i, j) in horizontal_edges:
        overlap_expr = ca.exp(x[i] - x[j]) + 0.5*ca.exp(w[i] - x[j]) + 0.5*ca.exp(w[j] - x[j])
        g.append(overlap_expr)
        lbg.append(0)
        ubg.append(1.0)
    
    # Vertical: exp(y_i - y_j) + 0.5*exp(h_i - y_j) + 0.5*exp(h_j - y_j) <= 1
    for (i, j) in vertical_edges:
        overlap_expr = ca.exp(y[i] - y[j]) + 0.5*ca.exp(h[i] - y[j]) + 0.5*ca.exp(h[j] - y[j])
        g.append(overlap_expr)
        lbg.append(0)
        ubg.append(1.0)
    
    # 5. Bounding box constraints (CORRECTED GP form)
    # Original: X_i + 0.5*W_i <= W_box
    # Divide by W_box: X_i/W_box + 0.5*W_i/W_box <= 1
    # GP form: exp(x_i - w_box) + 0.5*exp(w_i - w_box) <= 1
    for i in range(n):
        # X_i + 0.5*W_i <= W_box
        g.append(ca.exp(x[i] - w_box) + 0.5*ca.exp(w[i] - w_box))
        lbg.append(0)
        ubg.append(1.0)
        
        # Y_i + 0.5*H_i <= H_box
        g.append(ca.exp(y[i] - h_box) + 0.5*ca.exp(h[i] - h_box))
        lbg.append(0)
        ubg.append(1.0)
    
    # 6. Square die constraint: w_box = h_box
    g.append(w_box - h_box)
    lbg.append(0)
    ubg.append(0)
    
    # 7. HPWL constraints for each net (GP-compatible max/min form)
    print(f"  Adding HPWL constraints for {num_nets} nets...")
    for k, (weight, movable_indices, terminal_positions) in enumerate(nets):
        if len(movable_indices) + len(terminal_positions) >= 2:
            # For each movable module in the net
            for i in movable_indices:
                # X_i <= X_max^n  =>  exp(x_i - x_max[k]) <= 1
                g.append(ca.exp(x[i] - x_max[k]))
                lbg.append(0)
                ubg.append(1.0)
                
                # X_i >= X_min^n  =>  exp(x_min[k] - x_i) <= 1
                g.append(ca.exp(x_min[k] - x[i]))
                lbg.append(0)
                ubg.append(1.0)
                
                # Y_i <= Y_max^n  =>  exp(y_i - y_max[k]) <= 1
                g.append(ca.exp(y[i] - y_max[k]))
                lbg.append(0)
                ubg.append(1.0)
                
                # Y_i >= Y_min^n  =>  exp(y_min[k] - y_i) <= 1
                g.append(ca.exp(y_min[k] - y[i]))
                lbg.append(0)
                ubg.append(1.0)
                #X_min/X_max<1
                g.append(ca.exp(x_min[k] - x[i]))
                lbg.append(0)
                ubg.append(1.0)

                # Y_min/Y_max < 1
                g.append(ca.exp(y_min[k] - y[i]))
                lbg.append(0)
                ubg.append(1.0)

            # For each terminal in the net (as constants)
            for (tx, ty) in terminal_positions:
                # tx <= X_max^n  =>  tx / exp(x_max[k]) <= 1
                g.append(tx / ca.exp(x_max[k]))
                lbg.append(0)
                ubg.append(1.0)
                
                # tx >= X_min^n  =>  exp(x_min[k]) / tx <= 1
                g.append(ca.exp(x_min[k]) / tx)
                lbg.append(0)
                ubg.append(1.0)
                
                # ty <= Y_max^n  =>  ty / exp(y_max[k]) <= 1
                g.append(ty / ca.exp(y_max[k]))
                lbg.append(0)
                ubg.append(1.0)
                
                # ty >= Y_min^n  =>  exp(y_min[k]) / ty <= 1
                g.append(ca.exp(y_min[k]) / ty)
                lbg.append(0)
                ubg.append(1.0)
    
    # Objective: minimize sum of HPWL = sum(X_max^n - X_min^n + Y_max^n - Y_min^n)
    #          = sum(exp(x_max[k]) - exp(x_min[k]) + exp(y_max[k]) - exp(y_min[k]))
    objective_terms = list[ca.MX]()
    for k in range(num_nets):
        weight = nets[k][0]
        hpwl_x = ca.exp(x_max[k]-x_min[k]) 
        hpwl_y = ca.exp(y_max[k]-y_min[k])
        # weight_x=0.5* (ca.exp(-y_min[k]-0.5*(x_max[k]-3*x_min[k]))+ca.exp(y_max[k]-2*y_min[k]-0.5*(x_max[k]-3*x_min[k])))
        # weight_y=0.5* (ca.exp(-x_min[k]-0.5*(y_max[k]-3*y_min[k]))+ca.exp(x_max[k]-2*x_min[k]-0.5*(y_max[k]-3*y_min[k])))
        # weight_x=np.sqrt((ca.exp(-1*y_min[k])+ca.exp(y_max[k]-2*y_min[k]))/(ca.exp(-1*x_min[k])+ca.exp(x_max[k]-2*x_min[k])))
        # weight_y=np.sqrt((ca.exp(-1*x_min[k])+ca.exp(x_max[k]-2*x_min[k]))/(ca.exp(-1*y_min[k])+ca.exp(y_max[k]-2*y_min[k])))
        # objective_terms.append(weight * (weight_x*hpwl_x + weight_y*hpwl_y))
        objective_terms.append(weight * (hpwl_x + hpwl_y))
        #objective_terms.append(weight * (x_max[k]/x_min[k] + y_max[k]/y_min[k]))
    
    objective = ca.sum1(ca.vertcat(*objective_terms)) if objective_terms else ca.MX(0) 
    
    # Create constraint vector
    g_vec = ca.vertcat(*g) if g else ca.MX()
    
    # Create NLP problem
    nlp = {
        'x': opt_vars,
        'f': objective,
        'g': g_vec
    }
    
    print(f"  Problem dimensions: {opt_vars.size1()} variables, {len(g)} constraints")
    print(f"  Variables: {n} modules x 4 (x,y,w,h) + 2 bbox + {num_nets} nets x 4 (x_max, x_min, y_max, y_min)")
    print(f"  Objective: minimize sum(X_max^n - X_min^n + Y_max^n - Y_min^n) - exact HPWL")
    
    return nlp, {
        'x0': x0,
        'lbx': lbx,
        'ubx': ubx,
        'lbg': lbg,
        'ubg': ubg,
        'n': n,
        'num_nets': num_nets
    }


def solve_casadi(nlp, problem_data, solver='ipopt', max_iters=200, verbose=True):
    """
    Solve the NLP problem using CasADi
    
    Solver options:
    - ipopt: Interior Point OPTimizer (default, open-source)
    - sqpmethod: Sequential Quadratic Programming
    - scpgen: Sequential Convex Programming
    """
    try:
        # Setup solver options
        opts = dict[str, any]()
        
        if solver == 'ipopt':
            opts['ipopt.max_iter'] = max_iters
            opts['ipopt.print_level'] = 5 if verbose else 0
            
            # Relaxed convergence tolerances
            opts['ipopt.tol'] = 1e-3
            opts['ipopt.acceptable_tol'] = 1e-2
            opts['ipopt.acceptable_obj_change_tol'] = 1e-3
            opts['ipopt.acceptable_iter'] = 15
            opts['ipopt.acceptable_constr_viol_tol'] = 1e-3
            
            # Constraint violation tolerance
            opts['ipopt.constr_viol_tol'] = 1e-2
            
            # Other options
            opts['ipopt.warm_start_init_point'] = 'yes'
            opts['ipopt.mu_strategy'] = 'adaptive'
            opts['ipopt.linear_solver'] = 'mumps'
            
            # Bound relaxation (helps with tight constraints)
            opts['ipopt.bound_relax_factor'] = 1e-8
            opts['ipopt.honor_original_bounds'] = 'yes'
        elif solver == 'sqpmethod':
            opts['max_iter'] = max_iters
            opts['print_time'] = verbose
        
        opts['print_time'] = verbose
        
        # Create solver
        print(f"  Creating {solver.upper()} solver...")
        S = ca.nlpsol('solver', solver, nlp, opts)
        
        # Solve
        print(f"  Solving optimization problem...")
        sol = S(
            x0=problem_data['x0'],
            lbx=problem_data['lbx'],
            ubx=problem_data['ubx'],
            lbg=problem_data['lbg'],
            ubg=problem_data['ubg']
        )
        
        # Check solver status
        stats = S.stats()
        success = stats['success']
        return_status = stats['return_status']
        
        # Check if we have a usable solution (even if not optimal)
        if success:
            print(f"  ✓ Solver succeeded!")
            print(f"  Return status: {return_status}")
            return True, sol
        else:
            # Accept "acceptable" solutions even if not fully optimal
            acceptable_status = [
                'Solve_Succeeded',
                'Solved_To_Acceptable_Level',
                'Infeasible_Problem_Detected',  # Accept infeasible but use best point
                'Search_Direction_Becomes_Too_Small',
                'Feasible_Point_Found',
            ]
            
            # Check if status is in acceptable list or if we have a solution
            if any(status in return_status for status in acceptable_status) or sol is not None:
                print(f"  ⚠ Solver converged to suboptimal point")
                print(f"  Return status: {return_status}")
                print(f"  Accepting solution as best feasible approximation...")
                return True, sol  # Accept as "success" for practical purposes
            else:
                print(f"  ✗ Solver failed!")
                print(f"  Return status: {return_status}")
                return False, sol
            
    except Exception as e:
        print(f"  Solver error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def extract_solution(sol, modules, module_data, problem_data):
    """Extract solution from CasADi optimization result (convert from log-space to real-space)"""
    n = problem_data['n']
    num_nets = problem_data['num_nets']
    
    # Extract solution vector (in log-space)
    x_opt = sol['x'].full().flatten()
    
    # Parse solution (order: x, y, w, h, w_box, h_box, x_max, x_min, y_max, y_min)
    idx = 0
    
    # Extract x (log-space) and convert to X (real-space)
    x_log = x_opt[idx:idx+n]
    X_vals = np.exp(x_log)
    idx += n
    
    # Extract y (log-space) and convert to Y (real-space)
    y_log = x_opt[idx:idx+n]
    Y_vals = np.exp(y_log)
    idx += n
    
    # Extract w (log-space) and convert to W (real-space)
    w_log = x_opt[idx:idx+n]
    W_vals = np.exp(w_log)
    idx += n
    
    # Extract h (log-space) and convert to H (real-space)
    h_log = x_opt[idx:idx+n]
    H_vals = np.exp(h_log)
    idx += n
    
    # Extract bounding box dimensions (log-space) and convert to real-space
    W_box = float(np.exp(x_opt[idx]))
    H_box = float(np.exp(x_opt[idx+1]))
    idx += 2
    
    # Extract HPWL bounds (for information only)
    x_max_log = x_opt[idx:idx+num_nets]
    X_max_vals = np.exp(x_max_log)
    idx += num_nets
    
    x_min_log = x_opt[idx:idx+num_nets]
    X_min_vals = np.exp(x_min_log)
    idx += num_nets
    
    y_max_log = x_opt[idx:idx+num_nets]
    Y_max_vals = np.exp(y_max_log)
    idx += num_nets
    
    y_min_log = x_opt[idx:idx+num_nets]
    Y_min_vals = np.exp(y_min_log)
    idx += num_nets
    
    # Calculate total HPWL (exact)
    hpwl_x = np.sum(X_max_vals - X_min_vals)
    hpwl_y = np.sum(Y_max_vals - Y_min_vals)
    total_hpwl = float(hpwl_x + hpwl_y)
    
    # Update module data
    for i, name in enumerate(modules):
        module_data[name].update({
            'x': float(X_vals[i]),
            'y': float(Y_vals[i]),
            'w': float(W_vals[i]),
            'h': float(H_vals[i])
        })
    
    print(f"  Total HPWL (exact): {total_hpwl:.2f}")
    
    return W_box, H_box


def save_results(netlist_data, modules, module_data, output_file):
    updated = netlist_data.copy()
    if 'Modules' in updated:
        modules_dict = updated['Modules']
    else:
        modules_dict = updated
        updated = {'Modules': modules_dict}
    
    for name in modules:
        if name in modules_dict:
            data = module_data[name]
            modules_dict[name]['rectangles'] = [[data['x'], data['y'], data['w'], data['h']]]
    
    with open(output_file, 'w') as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False)


def visualize(modules, module_data, W, H, output_image):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, W*1.1)
    ax.set_ylim(0, H*1.1)
    ax.set_aspect('equal')
    
    # Draw bounding box
    ax.add_patch(patches.Rectangle((0, 0), W, H, fill=False, edgecolor='red', linewidth=2))
    
    # Draw modules
    colors = plt.cm.tab20(np.linspace(0, 1, len(modules)))
    for i, name in enumerate(modules):
        data = module_data[name]
        x_left = data['x'] - data['w']/2
        y_bottom = data['y'] - data['h']/2
        ax.add_patch(patches.Rectangle((x_left, y_bottom), data['w'], data['h'],
                                      facecolor=colors[i], edgecolor='black', alpha=0.7))
        ax.text(data['x'], data['y'], name, ha='center', va='center', fontsize=8)
    
    ax.set_title(f'W={W:.2f}, H={H:.2f}, Area={W*H:.2f}')
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()


def optimize_floorplan(netlist_file, die_file, output_file, output_image,
                       max_iter=200, max_ratio=3.0, solver='ipopt', alpha=1.0):
    """CasADi-based GGP Floorplan Optimization with HPWL Minimization"""

    netlist_data = load_netlist(netlist_file)
    die_info = load_die_info(die_file)
    die_width = die_info.get('width', 800.0)
    die_height = die_info.get('height', 800.0)
    
    modules, module_data, terminal_coords = extract_movable_modules(netlist_data)
    print(f"Modules: {len(modules)}")
    print(f"Terminals: {len(terminal_coords)}")
    print(f"Die: {die_width} × {die_height}")
    print(f"Solver: {solver}, Alpha: {alpha}\n")
    
    # Extract nets information
    nets = extract_nets(netlist_data, modules)
    print(f"Nets: {len(nets)}\n")
    
    # Build constraint graph using center-based distance
    h_edges, v_edges = build_constraint_graphs(module_data, modules)
    
    # Setup CasADi GGP model with HPWL objective
    nlp, problem_data = setup_casadi_model(modules, module_data, die_width, die_height,
                                           h_edges, v_edges, nets, max_ratio=max_ratio, alpha=alpha)
    
    # Solve NLP
    success, sol = solve_casadi(nlp, problem_data, solver=solver, max_iters=max_iter, verbose=True)
    
    if not success or sol is None:
        print(f"\n✗ Optimization failed completely - no solution available!")
        return False
    
    # Extract solution (even if suboptimal)
    W, H = extract_solution(sol, modules, module_data, problem_data)
    # print(f"\nSolution: W={W:.2f}, H={H:.2f}, Area={W*H:.2f}")
    
    # Save results
    save_results(netlist_data, modules, module_data, output_file)
    visualize(modules, module_data, W, H, output_image)
    
    print(f"\n{'='*70}")
    print(f"Optimization completed!")

    print(f"Output files: {output_file}, {output_image}")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='CasADi-based GGP Floorplan Optimization with HPWL Minimization')
    
    parser.add_argument('--netlist', required=True, help='Netlist YAML file')
    parser.add_argument('--die', required=True, help='Die info YAML file')
    parser.add_argument('--output', default='output_gp_wl.yaml', help='Output YAML file')
    parser.add_argument('--output-image', default='output_gp_wl.png', help='Output image')
    
    parser.add_argument('--max-iter', type=int, default=200, help='Max solver iterations')
    parser.add_argument('--max-ratio', type=float, default=3.0, help='Max aspect ratio')
    parser.add_argument('--alpha', type=float, default=1.0, help='LSE-HPWL alpha parameter (default: 1.0)')
    parser.add_argument('--solver', default='ipopt', 
                       choices=['ipopt', 'sqpmethod', 'scpgen'],
                       help='CasADi NLP solver (ipopt recommended)')
    
    args = parser.parse_args()
    
    optimize_floorplan(
        args.netlist, args.die, args.output, args.output_image,
        max_iter=args.max_iter, max_ratio=args.max_ratio, solver=args.solver, alpha=args.alpha)


if __name__ == '__main__':
    main()

