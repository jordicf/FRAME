# (c) Antoni Pech Alberich 2024
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import argparse
from typing import Any
try:
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch is not installed. To use this tool it is a tool requirment")

from tools.floorset_parser.floor_set_manager.manager import FloorSetInstance
from tools.floorset_parser.floor_set_manager.loaders.lite import FloorplanDatasetLite
from tools.floorset_parser.floor_set_manager.loaders.prime import FloorplanDatasetPrime
from tools.floorset_parser.floor_set_manager.loaders.collate import floorplan_collate, floorplan_collate_lite


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
        description="A FloorplanSet handler.",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "--input",
        default= './',
        type=str,
        help="Path to the location of the data. If data is not present, it will be downloaded. Default ./"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["PrimeTraining", "PrimeTest", "LiteTraining", "LiteTest"],
        help="Choose one of: PrimeTraining, PrimeTest, LiteTraining, LiteTest"
    )
    parser.add_argument(
        "--output",
        type=str,
        default='./',
        help="Destination folder of the Die YAML output file."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d",
        "--connection_density",
        type=float,
        default= None,
        help="Percentage for connection density between 0 and 1 (default: No changes)."
    )
    group.add_argument(
        "-f",
        "--scale_factor",
        type=float,
        default= None,
        help="Scaling factor for net weights (default: No changes)."
    )
    parser.add_argument(
        "--store-terminals",
        action="store_true",
        help="Store terminals as fixed modules instead of using a terminal flag"
    )
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    datasettype: str = options['dataset']
    inputdatapath: str = options['input']
    outfilepath: str = options['output']
    density: float|None = options['connection_density']
    factor: float|None = options['scale_factor']
    term2mod:bool = options['store_terminals']

    ds: Any
    if datasettype == "PrimeTraining":
        ds = FloorplanDatasetPrime(inputdatapath)
        fn = floorplan_collate
        bs = 512
        info = 'Processing FloorSet-Prime Batches'
    elif datasettype == "PrimeTest":
        ds = FloorplanDatasetPrime(inputdatapath, validation=True)
        fn = floorplan_collate
        bs = 1
        info = 'Processing FloorSet-Prime Test Batches'
    elif datasettype == "LiteTraining":
        ds = FloorplanDatasetLite(inputdatapath)
        fn = floorplan_collate_lite
        bs = 128
        info = 'Processing FloorSet-Lite Batches'
    elif datasettype == "LiteTest":
        ds = FloorplanDatasetLite(inputdatapath, validation=True)
        fn = floorplan_collate
        bs = 1
        info = 'Processing FloorSet-Lite Test Batches'
    else:
        raise ValueError(f"Wrong dataset key {datasettype} is not an option. Try -help to see possible options")

    dl = DataLoader(
        ds, 
        batch_size=bs, 
        shuffle=False,
        collate_fn=fn
    )

    for batch_idx, batch in enumerate(dl):
        
        area_targets, b2b_connectivities, p2b_connectivities, pins_positions, placement_constraints = batch[0]
        if (len(batch[1]) == 3):
            b_trees, solutions, metrics_list = batch[1]
        else:
            solutions, metrics_list = batch[1]

        for i in range(len(area_targets)): # Extract each floorplan
            curr_id = batch_idx * len(area_targets) + i

            # Preprocess tensors to remove invalid data (-1)
            area_target = area_targets[i][area_targets[i] != -1].numpy()
            b2b_connectivity = b2b_connectivities[i][b2b_connectivities[i][:, 0] != -1].numpy()
            p2b_connectivity = p2b_connectivities[i][p2b_connectivities[i][:, 0] != -1].numpy()
            pins_pos = pins_positions[i][pins_positions[i][:, 0] != -1].numpy()
            placement_constraint = placement_constraints[i][placement_constraints[i][:, 0] != -1].numpy() # nblocks x 5
            solution = solutions[i].numpy()
            metrics = metrics_list[i][metrics_list[i] != -1].numpy()

            data = {
                'area_blocks': area_target,
                'b2b_connectivity': b2b_connectivity,
                'p2b_connectivity': p2b_connectivity,
                'pins_pos': pins_pos,
                'placement_constraints': placement_constraint,
                'vertex_blocks': solution,
                'metrics': metrics
            } #TODO b_tree only for Lite Training

            fp = FloorSetInstance(data, density, factor, term2mod)

            # Create a filename for the item
            filename = f"_{curr_id}"
            fp.write_yaml_FPEF(outfilepath + "FPEF" + filename + ".yaml")
            fp.write_yaml_DIEF(outfilepath + "DIEF" + filename + ".yaml")

    return


if __name__ == "__main__":
    main()
