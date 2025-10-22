#!/usr/bin/env python3
"""
GP-based Floorplan Optimization with Constraint Relaxation Strategy
Iteratively remove redundant constraints based on slack analysis
"""

import yaml
import numpy as np
from gekko import GEKKO
import os
os.environ.setdefault('LC_ALL', 'C.UTF-8')
os.environ.setdefault('LANG', 'C.UTF-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse


def load_netlist(netlist_file):
    with open(netlist_file, 'r') as f:
        return yaml.safe_load(f)


def load_die_info(die_file):
    with open(die_file, 'r') as f:
        return yaml.safe_load(f)


def extract_movable_modules(netlist_data):
    modules = []
    module_data = {}
    
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



def build_constraint_graphs(module_data, modules):
    """
    build initial constraint graph based on overlap detection
    """
    n = len(modules)
    horizontal_edges = []
    vertical_edges = []
    
    stats = {'no_overlap': 0, 'h_overlap': 0, 'v_overlap': 0, 'both_overlap': 0}
    
    for i in range(n):
        for j in range(i+1, n):
            data_i = module_data[modules[i]]
            data_j = module_data[modules[j]]
            
            xi_left = data_i['x'] - 0.5 * data_i['w']
            xi_right = data_i['x'] + 0.5 * data_i['w']
            yi_bottom = data_i['y'] - 0.5 * data_i['h']
            yi_top = data_i['y'] + 0.5 * data_i['h']
            
            xj_left = data_j['x'] - 0.5 * data_j['w']
            xj_right = data_j['x'] + 0.5 * data_j['w']
            yj_bottom = data_j['y'] - 0.5 * data_j['h']
            yj_top = data_j['y'] + 0.5 * data_j['h']
            

            h_overlap = max(0, min(xi_right, xj_right) - max(xi_left, xj_left))
            v_overlap = max(0, min(yi_top, yj_top) - max(yi_bottom, yj_bottom))
            
         
            h_edge = (i, j) if data_i['x'] <= data_j['x'] else (j, i)
            v_edge = (i, j) if data_i['y'] <= data_j['y'] else (j, i)
            

            if h_overlap <= 1e-6 and v_overlap <= 1e-6:
                # no overlap in both directions: add to both graphs
                horizontal_edges.append(h_edge)
                vertical_edges.append(v_edge)
                stats['no_overlap'] += 1
            elif h_overlap <= 1e-6 and v_overlap > 1e-6:
                # only vertical overlap: add to HCG
                horizontal_edges.append(h_edge)
                stats['v_overlap'] += 1
            elif h_overlap > 1e-6 and v_overlap <= 1e-6:
                # only horizontal overlap: add to VCG
                vertical_edges.append(v_edge)
                stats['h_overlap'] += 1
            else:
                # both overlap: compare overlap size
                if h_overlap > v_overlap:
                    # horizontal overlap is larger: add to VCG
                    vertical_edges.append(v_edge)
                else:
                    # vertical overlap is larger or equal: add to HCG
                    horizontal_edges.append(h_edge)
                stats['both_overlap'] += 1
    #for debug
    # print(f"  Constraint statistics:")
    # print(f"    No overlap (both H&V): {stats['no_overlap']} pairs")
    # print(f"    V overlap only (→HCG): {stats['v_overlap']} pairs")
    # print(f"    H overlap only (→VCG): {stats['h_overlap']} pairs")
    # print(f"    Both overlap: {stats['both_overlap']} pairs")
    # print(f"  Total constraints: H={len(horizontal_edges)}, V={len(vertical_edges)}")
    
    return horizontal_edges, vertical_edges


def setup_gp_model(modules, module_data, die_width, die_height,
                   horizontal_edges, vertical_edges, max_ratio=3.0):
    """GGP Model Setup"""
    n = len(modules)
    g = GEKKO(remote=False)
    g.options.SOLVER = 1
    
    # create variables
    Xs, Ys, Ws, Hs, A = [], [], [], [], []
    
    for module_name in modules:
        data = module_data[module_name]
        
        x = g.Var(lb=1e-3, ub=die_width)
        y = g.Var(lb=1e-3, ub=die_height)
        w = g.Var(lb=1e-3, ub=die_width)
        h = g.Var(lb=1e-3, ub=die_height)
        
        x.value = [data['x']]
        y.value = [data['y']]
        w.value = [data['w']]
        h.value = [data['h']]
        
        Xs.append(x)
        Ys.append(y)
        Ws.append(w)
        Hs.append(h)
        A.append(data['area'])
    
    # bounding box variables
    W = g.Var(lb=0.5, ub=2*die_width)#lb can be adjusted based on initial solution
    H = g.Var(lb=0.5, ub=2*die_height)


    current_W = max(module_data[modules[i]]['x'] + 0.5 * module_data[modules[i]]['w'] for i in range(n))
    current_H = max(module_data[modules[i]]['y'] + 0.5 * module_data[modules[i]]['h'] for i in range(n))
    W.value = [current_W]
    H.value = [current_H]



    # constraints
    # 1. area constraint
    for i in range(n):
        g.Equation((Ws[i] * Hs[i]) / A[i] == 1)
    
    # 2. aspect ratio constraint
    for i in range(n):
        g.Equation(Ws[i] / (Hs[i] * max_ratio) <= 1)
        g.Equation(Hs[i] / (Ws[i] * max_ratio) <= 1)
    
    # 3. left bound
    for i in range(n):
        g.Equation(0.5 * Ws[i] / Xs[i] <= 1)
        g.Equation(0.5 * Hs[i] / Ys[i] <= 1)
    
    # 4. no overlap constraints 
    print(f"  Adding no-overlap constraints: H={len(horizontal_edges)}, V={len(vertical_edges)}")
    
    for (i, j) in horizontal_edges:
        g.Equation((Xs[i] + 0.5*Ws[i] + 0.5*Ws[j]) / Xs[j] <= 1)
    
    for (i, j) in vertical_edges:
        g.Equation((Ys[i] + 0.5*Hs[i] + 0.5*Hs[j]) / Ys[j] <= 1)
    
    # 5. bounding box constraints
    for i in range(n):
        g.Equation((Xs[i] + 0.5*Ws[i]) / W <= 1)
        g.Equation((Ys[i] + 0.5*Hs[i]) / H <= 1)
    
    # 6. aspect ratio constraint(optional)
    g.Equation(W / H == 1)
    
    # 7. objective function
    g.Obj(W)
    
    return g, {'Xs': Xs, 'Ys': Ys, 'Ws': Ws, 'Hs': Hs, 'W': W, 'H': H, 'n': n}


def solve_gp(g, max_iter=200):
    g.options.DIAGLEVEL = 0
    g.options.MAX_ITER = max_iter
    g.options.COLDSTART = 1
    g.options.REDUCE = 3
    g.options.OTOL = 1e-1
    g.options.RTOL = 1e-1
    
    try:
        g.solve(disp=False)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def calculate_constraint_slacks(variables, horizontal_edges, vertical_edges):
    """
    calculate slack value for each constraint in no overlap
    slack = x_j - (x_i + 0.5*w_i + 0.5*w_j) or x_j-xi?
    """
    Xs = variables['Xs']
    Ys = variables['Ys']
    Ws = variables['Ws']
    Hs = variables['Hs']
    
    def get_val(var):
        return var.value[0] if isinstance(var.value, list) else var.value
    
    h_slacks = {}
    for (i, j) in horizontal_edges:
        xi, xj = get_val(Xs[i]), get_val(Xs[j])
        wi, wj = get_val(Ws[i]), get_val(Ws[j])
        #slack = xj - (xi + 0.5*wi + 0.5*wj)
        slack = xj - xi
        h_slacks[(i, j)] = slack
    
    v_slacks = {}
    for (i, j) in vertical_edges:
        yi, yj = get_val(Ys[i]), get_val(Ys[j])
        hi, hj = get_val(Hs[i]), get_val(Hs[j])
        #slack = yj - (yi + 0.5*hi + 0.5*hj)
        slack = yj - yi
        v_slacks[(i, j)] = slack
    
    return h_slacks, v_slacks


def remove_redundant_constraints(horizontal_edges, vertical_edges, h_slacks, v_slacks, k=None):
    """
    Constraint Relaxation Strategy
    
    for module pairs with both H and V constraints,
    - keep the constraint with larger slack
    - remove the constraint with smaller slack (more likely to be active)
    
    Args:
        k: if specified, only remove the top k constraints with largest slack
    """
    # find module pairs with both H and V constraints
    h_pairs = {tuple(sorted(edge)) for edge in horizontal_edges}
    v_pairs = {tuple(sorted(edge)) for edge in vertical_edges}
    common_pairs = h_pairs & v_pairs
    
    if len(common_pairs) == 0:
        print("  No common pairs to relax")
        return horizontal_edges, vertical_edges, 0
    
    pair_info = []
    for pair in common_pairs:
        i, j = pair
        h_edge = (i, j) if (i, j) in h_slacks else (j, i)
        v_edge = (i, j) if (i, j) in v_slacks else (j, i)
        
        slack_h = h_slacks.get(h_edge, 0)
        slack_v = v_slacks.get(v_edge, 0)
        
        pair_info.append((pair, h_edge, v_edge, slack_h, slack_v, max(slack_h, slack_v)))
    
    # sort by max slack (smooth relaxation strategy)
    pair_info.sort(key=lambda x: x[5], reverse=True)
    
    # limit the number of constraints to remove
    if k is not None:
        pair_info = pair_info[:k]
    
    # remove constraints
    h_to_remove = set()
    v_to_remove = set()
    
    for pair, h_edge, v_edge, slack_h, slack_v, _ in pair_info:
        if slack_h >= slack_v:
            # keep H, remove V
            v_to_remove.add(v_edge)
        else:
            # keep V, remove H
            h_to_remove.add(h_edge)
    
    new_h = [e for e in horizontal_edges if e not in h_to_remove]
    new_v = [e for e in vertical_edges if e not in v_to_remove]
    
    removed = len(h_to_remove) + len(v_to_remove)
    
    print(f"  Relaxed {removed} constraints (H: -{len(h_to_remove)}, V: -{len(v_to_remove)})")
    print(f"  Remaining: H={len(new_h)}, V={len(new_v)}")
    
    return new_h, new_v, removed


def extract_solution(variables, modules, module_data):
    def get_val(var):
        return var.value[0] if isinstance(var.value, list) else var.value
    
    for i, name in enumerate(modules):
        module_data[name].update({
            'x': float(get_val(variables['Xs'][i])),
            'y': float(get_val(variables['Ys'][i])),
            'w': float(get_val(variables['Ws'][i])),
            'h': float(get_val(variables['Hs'][i]))
        })
    
    W = float(get_val(variables['W']))
    H = float(get_val(variables['H']))
    
    return W, H


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
    
 
    ax.add_patch(patches.Rectangle((0, 0), W, H, fill=False, edgecolor='red', linewidth=2))
    

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


def optimize_with_relaxation(netlist_file, die_file, output_file, output_image,
                             max_iter=200, max_ratio=3.0, max_rounds=10,
                             output_dir="output", k_per_round=None):
    """GP-based Floorplan Optimization with Constraint Relaxation"""

    netlist_data = load_netlist(netlist_file)
    die_info = load_die_info(die_file)
    die_width = die_info.get('width', 800.0)
    die_height = die_info.get('height', 800.0)
    
    modules, module_data = extract_movable_modules(netlist_data)
    print(f"Modules: {len(modules)}")
    print(f"Die: {die_width} × {die_height}\n")
    
    h_edges, v_edges = build_constraint_graphs(module_data, modules)
    

    os.makedirs(output_dir, exist_ok=True)
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*70}")
        print(f"Round {round_num}")
        print(f"{'='*70}")
        
        g, variables = setup_gp_model(modules, module_data, die_width, die_height,
                                       h_edges, v_edges, max_ratio=max_ratio)
        
        success = solve_gp(g, max_iter)
        
        if not success:
            print(f"\n Round {round_num} failed, stopping")
            break
        
        W, H = extract_solution(variables, modules, module_data)
        print(f"\n Solution: W={W:.2f}, H={H:.2f}, Area={W*H:.2f}")

        round_file = f"{output_dir}/round_{round_num}.yaml"
        round_img = f"{output_dir}/round_{round_num}.png"
        save_results(netlist_data, modules, module_data, round_file)
        visualize(modules, module_data, W, H, round_img)
        
        # Check Status
        h_slacks, v_slacks = calculate_constraint_slacks(variables, h_edges, v_edges)
        
        # remove redundant constraints
        new_h, new_v, removed = remove_redundant_constraints(
            h_edges, v_edges, h_slacks, v_slacks, k=k_per_round)
        
        if removed == 0:
            break
        
        # update constraint graph (next round)
        h_edges, v_edges = new_h, new_v
    

    save_results(netlist_data, modules, module_data, output_file)
    visualize(modules, module_data, W, H, output_image)
      

def main():
    parser = argparse.ArgumentParser(
        description='GP-based Floorplan Optimization with Constraint Relaxation')
    
    parser.add_argument('--netlist', required=True, help='Netlist YAML file')
    parser.add_argument('--die', required=True, help='Die info YAML file')
    parser.add_argument('--output', default='output.yaml', help='Output YAML file')
    parser.add_argument('--output-image', default='output.png', help='Output image')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    parser.add_argument('--max-iter', type=int, default=200, help='Max GP iterations')
    parser.add_argument('--max-ratio', type=float, default=3.0, help='Max aspect ratio')
    parser.add_argument('--max-rounds', type=int, default=10, help='Max relaxation rounds')
    parser.add_argument('--k-per-round', type=int, default=None,
                       help='Remove k constraints per round (None=remove all redundant)')
    
    args = parser.parse_args()
    
    optimize_with_relaxation(
        args.netlist, args.die, args.output, args.output_image,
        max_iter=args.max_iter, max_ratio=args.max_ratio,
        max_rounds=args.max_rounds, output_dir=args.output_dir,
        k_per_round=args.k_per_round)


if __name__ == '__main__':
    main()
