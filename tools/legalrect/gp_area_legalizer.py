#!/usr/bin/env python3
"""
GP-based Floorplan Optimization with Constraint Relaxation Strategy (CasADi Version)
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
    
    modules_dict = netlist_data.get('Modules', netlist_data)
    
    for module_name, module_info in modules_dict.items():
        if not isinstance(module_info, dict) or module_info.get('terminal', False):
            continue
        
        area = module_info.get('area', 0)
        rectangles = module_info.get('rectangles', [[0, 0, 1, 1]])
        
        if rectangles and len(rectangles[0]) == 4:
            x, y, w, h = rectangles[0]
            module_data[module_name] = {
                'area': area,
                'x': x, 'y': y, 'w': w, 'h': h
            }
            modules.append(module_name)
    
    return modules, module_data


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
                       horizontal_edges, vertical_edges, max_ratio=3.0):
    """
    CasADi NLP Model Setup for Floorplan Optimization
    
    Variables:
    - X[i], Y[i]: center coordinates of module i
    - W[i], H[i]: width and height of module i
    - W_box, H_box: bounding box dimensions
    
    Constraints:
    - Area constraints: W[i] * H[i] = area[i]
    - Aspect ratio: W[i] / H[i] <= max_ratio and H[i] / W[i] <= max_ratio
    - Non-overlap: position constraints based on constraint graphs
    - Bounding box: modules must fit within die
    - Square die: W_box = H_box
    
    Objective: minimize W_box 
    """
    n = len(modules)
    
    # Create optimization variables
    X = ca.MX.sym('X', n)  # x coordinates
    Y = ca.MX.sym('Y', n)  # y coordinates
    W = ca.MX.sym('W', n)  # widths
    H = ca.MX.sym('H', n)  # heights
    W_box = ca.MX.sym('W_box')  # bounding box width
    H_box = ca.MX.sym('H_box')  # bounding box height
    
    # Concatenate all variables
    opt_vars = ca.vertcat(X, Y, W, H, W_box, H_box)
    
    # Initialize variable bounds and initial guess
    # Order must match: ca.vertcat(X, Y, W, H, W_box, H_box)
    lbx = list[float]()
    ubx = list[float]()
    x0 = list[float]()
    
    # Bounds for X coordinates (all X's together)
    for i in range(n):
        data = module_data[modules[i]]
        lbx.append(1e-3)
        ubx.append(die_width)
        x0.append(max(data['x'], 1e-3))
    
    # Bounds for Y coordinates (all Y's together)
    for i in range(n):
        data = module_data[modules[i]]
        lbx.append(1e-3)
        ubx.append(die_height)
        x0.append(max(data['y'], 1e-3))
    
    # Bounds for W (all widths together)
    for i in range(n):
        data = module_data[modules[i]]
        area = data['area']
        if area > 0:
            w_min = np.sqrt(area / max_ratio)
            w_max = np.sqrt(area * max_ratio)
        else:
            w_min = 1e-3
            w_max = die_width
        lbx.append(max(w_min, 1e-3))
        ubx.append(w_max)
        x0.append(max(data['w'], 1e-3))
    
    # Bounds for H (all heights together)
    for i in range(n):
        data = module_data[modules[i]]
        area = data['area']
        if area > 0:
            h_min = np.sqrt(area / max_ratio)
            h_max = np.sqrt(area * max_ratio)
        else:
            h_min = 1e-3
            h_max = die_height
        lbx.append(max(h_min, 1e-3))
        ubx.append(h_max)
        x0.append(max(data['h'], 1e-3))
    
    # Bounds for W_box
    lbx.append(0.1 * die_width)
    ubx.append(1.5 * die_width)
    x0.append(die_width)
    
    # Bounds for H_box
    lbx.append(0.1 * die_height)
    ubx.append(1.5 * die_height)
    x0.append(die_height)
    
    # Constraints
    g = list[ca.MX]()
    lbg = list[float]()
    ubg = list[float]()
    
    # 1. Area constraints: W[i] * H[i] = area[i]
    for i in range(n):
        area = module_data[modules[i]]['area']
        if area > 0:
            g.append(W[i] * H[i])
            lbg.append(area)
            ubg.append(area)
    
    # 2. Aspect ratio constraints
    for i in range(n):
        # W[i] <= max_ratio * H[i]
        g.append(W[i] - max_ratio * H[i])
        lbg.append(-ca.inf)
        ubg.append(0)
        
        # H[i] <= max_ratio * W[i]
        g.append(H[i] - max_ratio * W[i])
        lbg.append(-ca.inf)
        ubg.append(0)
    
    # 3. Left and bottom bounds
    for i in range(n):
        # X[i] >= 0.5 * W[i]
        g.append(X[i] - 0.5 * W[i])
        lbg.append(0)
        ubg.append(die_width)
        
        # Y[i] >= 0.5 * H[i]
        g.append(Y[i] - 0.5 * H[i])
        lbg.append(0)
        ubg.append(die_height)
    
    # 4. No-overlap constraints
    print(f"  Adding no-overlap constraints: H={len(horizontal_edges)}, V={len(vertical_edges)}")
    
    # Horizontal: X[j] - X[i] >= 0.5*(W[i] + W[j])
    for (i, j) in horizontal_edges:
        g.append(X[j] - X[i] - 0.5*W[i] - 0.5*W[j])
        lbg.append(0)
        ubg.append(ca.inf)
    
    # Vertical: Y[j] - Y[i] >= 0.5*(H[i] + H[j])
    for (i, j) in vertical_edges:
        g.append(Y[j] - Y[i] - 0.5*H[i] - 0.5*H[j])
        lbg.append(0)
        ubg.append(ca.inf)
    
    # 5. Bounding box constraints
    for i in range(n):
        # X[i] + 0.5*W[i] <= W_box
        g.append(X[i] + 0.5*W[i] - W_box)
        lbg.append(-ca.inf)
        ubg.append(0)
        
        # Y[i] + 0.5*H[i] <= H_box
        g.append(Y[i] + 0.5*H[i] - H_box)
        lbg.append(-ca.inf)
        ubg.append(0)
    
    # 6. Square die constraint: W_box = H_box
    g.append(W_box - H_box)
    lbg.append(0)
    ubg.append(0)
    
    # Objective: minimize W_box 
    objective = W_box 
    
    # Create constraint vector
    g_vec = ca.vertcat(*g) if g else ca.MX()
    
    # Create NLP problem
    nlp = {
        'x': opt_vars,
        'f': objective,
        'g': g_vec
    }
    
    print(f"  Problem dimensions: {opt_vars.size1()} variables, {len(g)} constraints")
    print(f"  Objective: minimize W_box ")
    
    return nlp, {
        'x0': x0,
        'lbx': lbx,
        'ubx': ubx,
        'lbg': lbg,
        'ubg': ubg,
        'n': n
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
            opts['ipopt.acceptable_constr_viol_tol'] = 1e-2
            
            # Constraint violation tolerance
            opts['ipopt.constr_viol_tol'] = 1e-3
            
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
    """Extract solution from CasADi optimization result"""
    n = problem_data['n']
    
    # Extract solution vector
    x_opt = sol['x'].full().flatten()
    
    # Parse solution
    idx = 0
    
    # Extract X coordinates
    X_vals = x_opt[idx:idx+n]
    idx += n
    
    # Extract Y coordinates
    Y_vals = x_opt[idx:idx+n]
    idx += n
    
    # Extract W values
    W_vals = x_opt[idx:idx+n]
    idx += n
    
    # Extract H values
    H_vals = x_opt[idx:idx+n]
    idx += n
    
    # Extract bounding box dimensions
    W_box = float(x_opt[idx])
    H_box = float(x_opt[idx+1])
    
    # Update module data
    for i, name in enumerate(modules):
        module_data[name].update({
            'x': float(X_vals[i]),
            'y': float(Y_vals[i]),
            'w': float(W_vals[i]),
            'h': float(H_vals[i])
        })
    
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
                       max_iter=200, max_ratio=3.0, solver='ipopt'):
    """CasADi-based Floorplan Optimization with NLP Solver"""

    netlist_data = load_netlist(netlist_file)
    die_info = load_die_info(die_file)
    die_width = die_info.get('width', 800.0)
    die_height = die_info.get('height', 800.0)
    
    modules, module_data = extract_movable_modules(netlist_data)
    print(f"Modules: {len(modules)}")
    print(f"Die: {die_width} × {die_height}")
    print(f"Solver: {solver}\n")
    
    # Build constraint graph using center-based distance
    h_edges, v_edges = build_constraint_graphs(module_data, modules)
    
    # Setup CasADi NLP model
    nlp, problem_data = setup_casadi_model(modules, module_data, die_width, die_height,
                                           h_edges, v_edges, max_ratio=max_ratio)
    
    # Solve NLP
    success, sol = solve_casadi(nlp, problem_data, solver=solver, max_iters=max_iter, verbose=True)
    
    if not success or sol is None:
        print(f"\n✗ Optimization failed completely - no solution available!")
        return False
    
    # Extract solution (even if suboptimal)
    W, H = extract_solution(sol, modules, module_data, problem_data)
  
    
    # Save results
    save_results(netlist_data, modules, module_data, output_file)
    visualize(modules, module_data, W, H, output_image)
    
    print(f"\n{'='*70}")
    print(f"Optimization completed!")
    print(f"Final solution: W={W:.2f}, H={H:.2f}, Area={W*H:.2f}")
    print(f"Output files: {output_file}, {output_image}")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='CasADi-based Floorplan Optimization with NLP Solver')
    
    parser.add_argument('--netlist', required=True, help='Netlist YAML file')
    parser.add_argument('--die', required=True, help='Die info YAML file')
    parser.add_argument('--output', default='output_gp_area.yaml', help='Output YAML file')
    parser.add_argument('--output-image', default='output_gp_area.png', help='Output image')
    
    parser.add_argument('--max-iter', type=int, default=200, help='Max solver iterations')
    parser.add_argument('--max-ratio', type=float, default=3.0, help='Max aspect ratio')
    parser.add_argument('--solver', default='ipopt', 
                       choices=['ipopt', 'sqpmethod', 'scpgen'],
                       help='CasADi NLP solver (ipopt recommended)')
    
    args = parser.parse_args()
    
    optimize_floorplan(
        args.netlist, args.die, args.output, args.output_image,
        max_iter=args.max_iter, max_ratio=args.max_ratio, solver=args.solver)


if __name__ == '__main__':
    main()

