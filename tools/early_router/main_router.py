import argparse
import warnings
from typing import Any
import traceback
from tools.early_router.manager import file_manager


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

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ======== SOLVE COMMAND ========
    solve_parser = subparsers.add_parser("solve", help="Run the early global router on a floorplan")
    solve_parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the folder or file (.yaml) containing one input floorplan data per file."
    )
    solve_parser.add_argument("--output", required=True, type=str, help="Path to output folder to store results")
    solve_parser.add_argument(
        "--importance",
        nargs=3,
        metavar=('wire_lenght', 'module_crossing', 'via_usage'),
        type=float,
        default=[0.1, 0.2, 0.7],  # or require=True if you want it mandatory
        help="Three importance factors (must sum to 1): wirelength, module crossing, via usage"
    )
    solve_parser.add_argument("--ILP", action="store_true", help="If true, solve ILP (slower), otherwise relaxed LP (faster).")
    # ======== COMPARE COMMAND ========
    compare_parser = subparsers.add_parser("compare", help="Compare results from two solution folders")
    compare_parser.add_argument("--input1", required=True, type=str, help="First results folder to compare")
    compare_parser.add_argument("--input2", required=True, type=str, help="Second results folder to compare")

    # ======== ANALYZE COMMAND ========
    analyze_parser = subparsers.add_parser("analyze", help="Perform additional analysis")

    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_type", required=True)

    # Analyze hyperparameters
    hyper_parser = analyze_subparsers.add_parser("hyperparams", help="Analyze solver hyperparameters")
    hyper_parser.add_argument("--input", required=True, type=str, help="Input file or folder for analysis")
    hyper_parser.add_argument("--output", required=True, type=str, help="Path to output folder to store results")

    # Analyze layers
    layers_parser = analyze_subparsers.add_parser("layers", help="Analyze optimal number of routing layers")
    layers_parser.add_argument("--input", required=True, type=str, help="Input file or folder")
    layers_parser.add_argument("--output", required=True, type=str, help="Path to output folder to store results")
    layers_parser.add_argument(
        "--importance",
        nargs=3,
        metavar=('wire_lenght', 'module_crossing', 'via_usage'),
        type=float,
        default=[0.1, 0.2, 0.7],  # or require=True if you want it mandatory
        help="Three importance factors (must sum to 1): wirelength, module crossing, via usage"
    )

    # ======== COMMON OPTIONS (added to all subcommands) ========
    for subparser in [solve_parser, compare_parser, hyper_parser, layers_parser]:
        subparser.add_argument("--draw-nets", nargs="*", type=int, default=None,
                               help="List of net IDs to draw. If omitted, 3 random nets will be chosen.")
        subparser.add_argument("--draw-congestion", action="store_true", help="Draw congestion map")
        subparser.add_argument("--asap7", action="store_true", help="Enable ASAP7 bounding box preprocessing")
        subparser.add_argument(
            "--pitch_layers",
            nargs=2,
            type=float,
            metavar=("H_PITCH", "V_PITCH"),
            default=[1.0, 1.0],
            help="Specify the pitch for horizontal and vertical layers."
        )
        subparser.add_argument(
            "--reweight-nets-range", nargs=2, type=float, metavar=('LOWER', 'UPPER'),
            help="Specify the lower and upper bounds for the nets weight range"
        )
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    try:
        file_manager(options)
    except Exception as e:
        warnings.warn(f"Could not finish execution due to \n{traceback.print_exc()}", UserWarning)
        return
        # try:
        #     from tools.early_router.isdp_parser import parse_isdp_file, convert_to_hanangrid
        #     from tools.early_router.hanan import HananGrid
        #     from tools.early_router.build_model import FeedThrough
        #     isdp_data = parse_isdp_file(str(options['input']))
        #     data = convert_to_hanangrid(isdp_data)
        #     hg = HananGrid(data['HananCells'])
        #     ft = FeedThrough(hg, layers=data['Layers'])
        #     nets = list(data['Nets'].values())
        #     if isdp_data['capacity_adjustments']:
        #         ft.set_capacity_adjustments({(
        #                 (a['row'], a['column'], a['layer']), (a['target_row'],a['target_column'], a['target_layer'])
        #                 ): a['reduced_capacity'] for a in isdp_data['capacity_adjustments']})
        #     ft.add_nets(nets)
        #     ft.build()
        #     fwl,fmc,fvu = options['importance']
        #     ft.solve(f_wl=fwl,f_mc=fmc,f_vu=fvu)
        #     metrics = ft.metrics
        #     metrics['name'] = 'ISPD98'
        #     csv_filename = f"{str(options['output'])}/ispdmetrics.csv"

        # except Exception as e:
        #     warnings.warn(f"Could not finish execution due to \n{traceback.print_exc()}", UserWarning)

    return


if __name__ == "__main__":
    main()

