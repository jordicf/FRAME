from distinctipy import distinctipy
from pathlib import Path
from typing import Any
from frame.netlist.netlist import Netlist
from tools.draw.draw import get_floorplan_plot, calculate_bbox
from tools.early_router.draw import (
    draw_congestion,
    draw_solution2D,
    draw_solution3D,
    plot_net_distribution,
)
from tools.early_router.build_model import FeedThrough
from tools.early_router.ispd_parser import parse_ispd_file, convert_to_hanangrid
from tools.early_router.compare_solutions import compare_solution_files
import random
import time
import csv
import numpy as np
import math
from frame.netlist.netlist_types import NamedHyperEdge


def run_compare(options: dict[str, Any]) -> None:
    input1 = Path(options["input1"])
    input2 = Path(options["input2"])
    # Implement comparison logic
    print(f"Comparing results between {input1} and {input2}...")
    d: dict[int, int] = compare_solution_files(input1, input2)
    # draw_nets, draw_congestion, asap7?
    l = [k for k, v in d.items() if v > 20]
    return


def run_analyze(options: dict[str, Any]) -> None:
    analyze_type = options["analyze_type"]
    input_path = Path(options["input"])
    output_path = Path(options["output"])

    if input_path.is_file():
        file_path = input_path
    assert file_path.suffix.lower() == ".yaml"

    if analyze_type == "hyperparams":
        print(f"Analyzing hyperparameters using {input_path}")
        netlist = Netlist(str(file_path))  # too much time for ISPD files
        nets = netlist.edges
        ft = FeedThrough(netlist, **options)
        ft.add_nets(nets)
        ft.build()
        hyperparameter_analysis(ft, output_path, file_path.stem)
        return

    elif analyze_type == "layers":
        print(f"Analyzing routing layers using {input_path}")
        ft1, p1 = high_congestion_opt(file_path, options)
        ft2, p2 = low_congestion_opt(file_path, options, high=p1)

        fwl, fmc, fvu = options["importance"]
        ft1.solve(f_wl=fwl, f_mc=fmc, f_vu=fvu)
        ft2.solve(f_wl=fwl, f_mc=fmc, f_vu=fvu)

        if options["draw_congestion"]:
            netlist = Netlist(str(file_path))
            congest_map = draw_congestion(
                netlist, ft1.solution.congestion, ft1.hanan_graph
            )
            congest_map.save(f"{output_path}/{file_path.stem}map_HC.gif", quality=95)

            congest_map = draw_congestion(
                netlist, ft2.solution.congestion, ft2.hanan_graph
            )
            congest_map.save(f"{output_path}/{file_path.stem}map_LC.gif", quality=95)

        ft1.save(filepath=f"{output_path}/", filename=f"{file_path.stem}routesHC")
        ft2.save(filepath=f"{output_path}/", filename=f"{file_path.stem}routesLC")

        combined_metrics = [ft1.metrics, ft2.metrics]
        all_keys = {key for routed in combined_metrics for key in routed.keys()}
        csv_filename = f"{output_path}/{file_path.stem}layers_data.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(combined_metrics)
        return

    else:
        raise ValueError(f"Unknown analyze type: {analyze_type}")


def low_congestion_opt(file_path, options, low=0.0, high=152.0, eps=0.01, max_iter=100):
    # Binary search
    sol = None
    itr = 0
    # Binary search for the high-congestion
    while high - low > eps and itr < max_iter:
        itr += 1
        c = (low + high) / 2
        print(f"Trying pitch layer with pitch c = {c})...")

        options["pitch_layers"] = [c, c]
        netlist = Netlist(str(file_path))
        ft = FeedThrough(netlist, **options)
        nets = netlist.edges
        ft.add_nets(nets)
        ft.build()
        fwl, fmc, fvu = options["importance"]
        _, m = ft.solve(f_wl=fwl, f_mc=fmc, f_vu=fvu)

        if _ and round(m["mrd"], 6) < 1:
            best_c = c
            low = c  # try to find a smaller c
            sol = ft
        else:
            high = c  # increase c
            sol = sol

    if best_c is not None and sol is not None:
        print(
            f"NO congestion with pitch: {best_c}, and number of layers: {math.ceil(76 / best_c)}"
        )
        sol._m["opt_nlayers"] = math.ceil(76 / best_c)
        return sol, best_c
    else:
        print("No suitable pitch found.")
        return None


def high_congestion_opt(
    file_path, options, low=0.0, high=152.0, eps=0.01, max_iter=100
):
    # Binary search
    sol = None
    itr = 0
    # Binary search for the high-congestion
    while high - low > eps and itr < max_iter:
        itr += 1
        c = (low + high) / 2
        print(f"Trying pitch layer with pitch c = {c})...")

        options["pitch_layers"] = [c, c]
        netlist = Netlist(str(file_path))
        ft = FeedThrough(netlist, **options)
        nets = netlist.edges
        ft.add_nets(nets)

        if ft.build():
            best_c = c
            low = c  # try to find a smaller c
            sol = ft
        else:
            high = c  # increase c
            sol = sol

    if best_c is not None and sol is not None:
        sol._m["opt_nlayers"] = math.ceil(76 / best_c)
        print(
            f"Best working pitch found: {best_c}, with opt number of layers: {math.ceil(76 / best_c)}"
        )
        return sol, best_c
    else:
        print("No suitable pitch found.")
        return None


def hyperparameter_analysis(ft: FeedThrough, output: Path, filename: str):
    results = []
    data = []

    def evaluate(alpha, beta, data):
        gamma = 1.0 - alpha - beta
        if gamma <= 0:
            return None  # skip invalid combination

        _, m = ft.solve(f_wl=alpha, f_mc=beta, f_vu=gamma)
        if not _:
            return None
        data.append(ft.metrics.copy())
        wl, mc, vu = m["norm_fact"]
        delta_wl = round(m["total_wl"] / wl - 1, 4)
        delta_mc = round(m["module_crossings"] / mc - 1, 4)
        delta_vu = round(m["via_usage"] / vu - 1, 4)
        # print(f"factors {(round(alpha,3),round(beta,3),round(gamma,3))}\tcost:{(delta_wl + delta_mc + delta_vu)/3}")
        return m["obj_val"]
        # return (delta_wl + delta_mc + delta_vu)/3

    min_weight = 0.02  # or any threshold you consider reasonable
    # Step 1: Random Initialization
    num_initial = 10
    for _ in range(num_initial):
        while True:
            alpha = round(random.uniform(0.01, 0.98), 3)
            beta = round(random.uniform(0.01, 1.0 - alpha - 0.01), 3)
            gamma = round(1.0 - alpha - beta, 3)
            if (
                alpha >= min_weight and beta >= min_weight and gamma >= min_weight
            ):  # Avoiding numerical instability
                break
        cost = evaluate(alpha, beta, data)
        if cost is not None:
            results.append(((alpha, beta, gamma), cost))

    # Step 2: Greedy Improvement
    for _ in range(40):  # total 50 trials
        # pick best result so far
        (best_alpha, best_beta, _), _ = min(results, key=lambda x: x[1])

        # explore nearby points
        perturb = lambda x: max(min(x + np.random.normal(0, 0.05), 0.98), 0.01)
        while True:
            new_alpha = perturb(best_alpha)
            new_beta = perturb(best_beta)
            new_gamma = 1.0 - new_alpha - new_beta
            if (
                new_alpha >= min_weight
                and new_beta >= min_weight
                and new_gamma >= min_weight
            ):  # Avoiding numerical inestability
                break

        cost = evaluate(new_alpha, new_beta, data)
        if cost is not None:
            results.append(((new_alpha, new_beta, new_gamma), cost))

    # Show best result
    best_params, best_cost = min(results, key=lambda x: x[1])
    print("Best weights:", best_params)
    print("Best cost:", best_cost)

    csv_filename = f"{output}/{filename}hyper_data.csv"
    # Get the keys from the first dictionary to use as headers
    all_keys = {key for routed in data for key in routed.keys()}
    sorted_keys = sorted(all_keys)
    # Write data to CSV
    with open(csv_filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted_keys)
        writer.writeheader()  # Write column names
        writer.writerows(data)  # Write each row from the list of dicts
    return


def draw_nets(ft: FeedThrough, netlist: Netlist, output: Path, filename: str, options):
    # Draw the net weight distribution
    plot_net_distribution(ft, filepath=f"{output}/{filename}wdistr")
    if not options["draw_nets"]:
        # Random drawing 3 nets
        num2draw = 3
        random.seed(2025)
        net_ids = random.sample(range(len(ft._nets)), num2draw)
    else:
        net_ids = options["draw_nets"]
        num2draw = len(net_ids)
    colors = distinctipy.get_colors(num2draw, rng=0)
    colors = [
        (round(r * 255), round(g * 255), round(b * 255), 255) for (r, g, b) in colors
    ]  # Opacity 128?
    netid2color = {j: colors[i] for i, j in enumerate(net_ids)}
    routes = {
        key: ft.solution.routes[key] for key in net_ids if key in ft.solution.routes
    }

    routes_map = draw_solution2D(netlist, routes, ft.hanan_graph, netid2color)
    routes_map.save(f"{output}/{filename}image_routes.gif", quality=95)
    for net_id in routes.keys():
        draw_solution3D(
            netlist,
            routes[net_id],
            ft._nets[net_id],
            ft.hanan_graph,
            net_color=netid2color[net_id],
            filepath=f"{output}/{filename}route3d{net_id}",
        )


def run_solve(options: dict[str, Any]) -> None:
    input_path: Path = Path(options["input"])
    output: Path = Path(options["output"])

    if input_path.is_file():
        filename = input_path.stem
    assert input_path.suffix.lower() == ".yaml", "Not given a .yaml file for solving."

    line = "################################################"
    dashed = "-----------------------------------------------"
    print(f"{line}\n\t\tRouting: {filename} ...")
    start_time = time.perf_counter()

    netlist = Netlist(str(input_path))
    nets = netlist.edges
    ft = FeedThrough(netlist, **options)
    ft.add_nets(nets)
    # from frame.netlist.netlist_types import NamedHyperEdge
    # n = NamedHyperEdge(['M1', 'M3', 'M6'], 500)
    # ft.add_nets([n])
    ft.build()
    set_up_time = time.perf_counter()
    print(f"Build time: {set_up_time - start_time:.6f} seconds")

    fwl, fmc, fvu = options["importance"]
    ft.solve(f_wl=fwl, f_mc=fmc, f_vu=fvu)
    metrics = ft.metrics
    metrics["name"] = filename
    solve_time = time.perf_counter()
    print(f"Solving time: {solve_time - set_up_time:.6f} seconds")

    if ft.has_solution():
        ft.save(filepath=f"{output}/", filename=f"{filename}routes")

        print(
            f"{dashed}\nTotal WL={metrics['total_wl']}\n"
            + f"Total Module Crossing={metrics['module_crossings']}\n"
            + f"Total Via Usage={metrics['via_usage']}\n{dashed}"
        )

        if options["draw_congestion"]:
            # Image with all net connections
            die_shape = calculate_bbox(netlist)
            im = get_floorplan_plot(netlist, die_shape)
            im.save(f"{output}/{filename}image.gif", quality=95)

            congest_map = draw_congestion(
                netlist, ft.solution.congestion, ft.hanan_graph
            )
            congest_map.save(f"{output}/{filename}image_congestion_map.gif", quality=95)

        if not options["draw_nets"] is None:
            draw_nets(ft, netlist, output, filename, options)

    return metrics


def file_manager(options: dict[str, Any]):
    # if not file_path.suffix.lower() == ".yaml":
    #     ispd_data = parse_ispd_file(str(file_path))
    #     data = convert_to_hanangrid(ispd_data)
    #     hg = HananGrid(data['HananCells'])
    #     ft = FeedThrough(hg, layers=data['Layers'])
    #     nets = list(data['Nets'].values())
    #     if ispd_data['capacity_adjustments']:
    #         ft.set_capacity_adjustments({(
    #             (a['row'], a['column'], a['layer']), (a['target_row'],a['target_column'], a['target_layer'])
    #             ): a['reduced_capacity'] for a in ispd_data['capacity_adjustments']})
    #     ft.add_nets(nets)
    #     ft.build()

    command = options["command"]

    if command == "solve":
        input_path: Path = Path(options["input"])
        floorplans: list = []
        if input_path.is_file():
            floorplans.append(run_solve(options))
        elif input_path.is_dir():
            for file in input_path.iterdir():
                options["input"] = file
                floorplans.append(file_manager(options))

        csv_filename = f"{Path(options['output'])}/all_metrics.csv"
        # Get the keys from the first dictionary to use as headers
        all_keys = {key for floorplan in floorplans for key in floorplan.keys()}
        # Write data to CSV
        with open(csv_filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()  # Write column names
            writer.writerows(floorplans)  # Write each row from the list of dicts

    elif command == "compare":
        run_compare(options)

    elif command == "analyze":
        input_path: Path = Path(options["input"])
        if input_path.is_file():
            run_analyze(options)
        elif input_path.is_dir():
            for file in input_path.iterdir():
                options["input"] = file
                run_analyze(options)

    return
