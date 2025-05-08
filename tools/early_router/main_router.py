import argparse
import warnings
from distinctipy import distinctipy
from pathlib import Path
from typing import Any
from frame.netlist.netlist import Netlist
from tools.early_router.hanan import Layer
from tools.draw.draw import get_floorplan_plot, calculate_bbox
from tools.early_router.draw import draw_congestion, draw_solution2D, draw_solution3D, plot_net_distribution
from tools.early_router.build_model import FeedThrough
from tools.early_router.isdp_parser import parse_isdp_file, convert_to_hanangrid
from tools.early_router.hanan import HananGrid
import random
import time
import csv
import traceback
import numpy as np
import math


def file_manager(file_path:Path, options:dict[str, Any]):

    output = Path(options['output'])
    filename = file_path.stem
    line = '################################################'
    dashed = '-----------------------------------------------'
    print(f"{line}\n\t\tRouting: {filename} ...")
    start_time = time.perf_counter()

    best_c = None
    if options["optimize_nlayers"] and file_path.suffix.lower() == ".yaml":
        # Binary search
        low = 1.0       # Smallest possible c (avoid 0)
        high = 130.0    # Max c to try
        eps = 1       # Precision tolerance
        while high - low > eps:
            c = (low + high) / 2
            print(f"Trying pitch layer with pitch c = {c})...")

            options['layers'] = [Layer('H', pitch=c), Layer('V', pitch=c)]
            netlist = Netlist(str(file_path))
            ft = FeedThrough(netlist, **options)
            nets = netlist.edges
            ft.add_nets(nets)

            if ft.build(options["optimize_bbox"]):
                best_c = c
                low = c  # try to find a smaller c
            else:
                high = c  # increase c

        if best_c is not None:
            print(f"Best working pitch found: {best_c}, with number of layers: {math.ceil(76/best_c)}")
        else:
            print("No suitable pitch found.")

    else:
        if file_path.suffix.lower() == ".yaml":
            netlist = Netlist(str(file_path)) # too much time for ISDP files
            ft = FeedThrough(netlist, **options)
            nets = netlist.edges
            
            if options['draw_congestion']:
                # Image with all net connections
                die_shape = calculate_bbox(netlist)
                im = get_floorplan_plot(netlist, die_shape)
                im.save(f"{output}/{filename}image.gif", quality=95)
        else:
            isdp_data = parse_isdp_file(str(file_path))
            data = convert_to_hanangrid(isdp_data)
            hg = HananGrid(data['HananCells'])
            ft = FeedThrough(hg, layers=data['Layers'])
            nets = list(data['Nets'].values())
            if isdp_data['capacity_adjustments']:
                ft.set_capacity_adjustments({(
                    (a['row'], a['column'], a['layer']), (a['target_row'],a['target_column'], a['target_layer'])
                    ): a['reduced_capacity'] for a in isdp_data['capacity_adjustments']})

        ft.add_nets(nets)
        ##################################### 
        # from frame.netlist.netlist_types import HyperEdge
        # multiple_pins=0
        # n = HyperEdge([netlist.get_module(name) for name in ['M1', 'M3', 'M6']], 4)
        # ft.add_nets([n])
        ft.build(options["optimize_bbox"])

    set_up_time = time.perf_counter()
    print(f"Build time: {set_up_time - start_time:.6f} seconds")
    
    if options['importance_analysis']:
        results = []
        epsilon = 1e-2
        alpha_vals = np.linspace(epsilon, 1 - epsilon, 10)
        beta_vals = np.linspace(epsilon, 1 - epsilon, 10)
        param_values = [(round(alpha,3), round(beta,3), round(1 - alpha - beta,3)) for alpha in alpha_vals for beta in beta_vals if alpha + beta < 1-epsilon]

        for alpha, beta, gamma in param_values:
            ft.solve(f_wl=alpha,f_mc=beta,f_vu=gamma)
            results.append(ft.metrics.copy())

        csv_filename = f"{output}/{filename}analysis.csv"
        # Get the keys from the first dictionary to use as headers
        all_keys = {key for routed in results for key in routed.keys()}
        # Write data to CSV
        with open(csv_filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()  # Write column names
            writer.writerows(results)  # Write each row from the list of dicts
        return {}

    else:
        fwl,fmc,fvu = options['importance']
        ft.solve(f_wl=fwl,f_mc=fmc,f_vu=fvu)
        metrics = ft.metrics
        metrics['name'] = filename
        if best_c is not None:
            metrics['best_nlayers'] = math.ceil(76/best_c)
        solve_time = time.perf_counter()
        print(f"Solving time: {solve_time - set_up_time:.6f} seconds")
        
        if ft.has_solution():
            ft.save(filepath=f"{output}/", filename=f"{filename}routes")
            print(f"{dashed}\nTotal WL={metrics['total_wl']}\nTotal Module Crossing={metrics['module_crossings']}\nTotal Via Usage={metrics['via_usage']}\n{dashed}")
            
            if options['draw_congestion']:
                congest_map = draw_congestion(netlist, ft.solution.congestion, ft.hanan_graph)
                congest_map.save(f"{output}/{filename}image_congestion_map.gif", quality=95)

            if not options['draw_nets'] is None:
                # Draw the net weight distribution
                plot_net_distribution(ft, filepath=f"{output}/{filename}wdistr")
                if not options['draw_nets']:
                    # Random drawing 3 nets
                    num2draw = 3
                    random.seed(2025)
                    net_ids= random.sample(range(len(nets)), num2draw)
                else:
                    net_ids = options['draw_nets']
                    num2draw = len(net_ids)
                colors = distinctipy.get_colors(num2draw, rng=0)
                colors = [(round(r * 255), round(g * 255), round(b * 255), 255) for (r, g, b) in colors] # Opacity 128?
                netid2color = {j: colors[i] for i, j in enumerate(net_ids)}
                routes = {key: ft.solution.routes[key] for key in net_ids if key in ft.solution.routes}

                routes_map = draw_solution2D(netlist, routes, ft.hanan_graph, netid2color)
                routes_map.save(f"{output}/{filename}image_routes.gif", quality=95)
                for net_id in routes.keys():
                    draw_solution3D(netlist, routes[net_id], ft._nets[net_id], ft.hanan_graph, net_color=netid2color[net_id], filepath=f"{output}/{filename}route3d{net_id}")

                end_time = time.perf_counter()
                print(f"Drawing time: {end_time - solve_time:.6f} seconds")

    return metrics


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse command-line arguments for the FloorplanSet handler.

    Args:
        prog (str | None): The program name to display in help messages.
        args (list[str] | None): A list of arguments to parse (defaults to sys.argv).

    Returns:
        dict[str, Any]: A dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="An Early Global Routing router.",
        usage="%(prog)s [options]",
    )
    # Input file argument
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the folder of file (.yaml) containing one input floorplan data per file."
    )
    # output folder argument
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path folder to store data"
    )
    parser.add_argument(
        "--importance",
        nargs=3,
        metavar=('wire_lenght', 'module_crossing', 'via_usage'),
        type=float,
        default=[0.4, 0.3, 0.3],  # or require=True if you want it mandatory
        help="Three importance factors for wirelength, module interference, and via usage. Must be floats in [0,1] that sum to 1."
    )
    parser.add_argument("--importance-analysis", action="store_true", help="Analyse the the best importance factors")
    parser.add_argument(
        "--reweight-nets-range",
        type=float,
        nargs=2,
        metavar=('LOWER', 'UPPER'),
        help="Specify the lower and upper bounds for the nets weight range"
    )
    parser.add_argument(
        "--draw-nets",
        nargs="*",
        type=int,
        default=None,
        help="List of net IDs to draw. If omitted after the flag, 3 random nets will be chosen."
    )
    parser.add_argument("--draw-congestion", action="store_true", help="Draw the congestion map")
    parser.add_argument("--asap7", action="store_true", help="Find optinal pre-routing bounding box for all nets.")
    parser.add_argument("--optimize-bbox", action="store_true", help="Compute the optimal bounding box")
    parser.add_argument("--optimize-nlayers", action="store_true", help="Compute the optimal number of 76 pitch layers")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    input_path: Path = Path(options['input'])
    output_path = Path(options['output'])

    floorplans:list = []

    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    if input_path.is_file():
        file_path = input_path
        try:
            floorplans.append(file_manager(file_path, options))
        except Exception as e:
            warnings.warn(f"Could not process {file_path.name} due to \n{traceback.print_exc()}", UserWarning)
            
    elif input_path.is_dir():
        for file in input_path.iterdir():
            if file.is_file() and file.name.startswith("FPEF") and file.suffix.lower() == ".yaml":
                file_path = file
                ##############
                # if file.stem > 'FPEF_1.yaml':
                #     break
                ##############
                try:
                    floorplans.append(file_manager(file_path, options))
                except Exception as e:
                    warnings.warn(f"Could not process {file_path.name} due to \n{traceback.print_exc()}")
                    #traceback.print_exc()
    else:
        raise ValueError(f"Input path {input_path} does not exist or is invalid.")
    
    csv_filename = f"{output_path}/floorplan_metrics.csv"
    # Get the keys from the first dictionary to use as headers
    all_keys = {key for floorplan in floorplans for key in floorplan.keys()}
    # Write data to CSV
    with open(csv_filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_keys)
        writer.writeheader()  # Write column names
        writer.writerows(floorplans)  # Write each row from the list of dicts

    return


if __name__ == "__main__":
    main()

