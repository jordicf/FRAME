import argparse
import networkx as nx
import numpy as np
from .modified_fa2 import forceatlas2_layout
from frame.die.die import Die
from frame.netlist.netlist import Netlist
from frame.netlist.netlist_types import Module
from frame.geometry.geometry import Point, Shape
from typing import Any

def netlist_to_graph(
        netlist: Netlist
    ) -> tuple[nx.Graph, dict[int, np.ndarray]]:
    """
    Converts the netlist into a networkx Graph, which is the
    required object type for the fa2algorithm (for now).

    Since hyperedges are not supported in networkx, a dummy
    vertex is added for each of them.

    Params:
        netlist: the netlist to convert to a graph
        seed: random seed to generate centers for modules with no
              position
    """

    def bounding_box(module: Module) -> tuple[float, float]:
        """
        Returns the bounding box width & height of a module.
        If the shape of the module hasn't been defined yet,
        it assumes it is a square.
        """

        # check if shape is undefined
        if module.num_rectangles() == 0:
            area: float = module.area()
            width: float = np.sqrt(area)
            return (width, width) # square bounding box
        
        x_max = y_max = float('-inf')
        x_min = y_min = float('inf')
        
        for rectangle in module._rectangles:
            shape : Shape = rectangle._shape
            center : Point = rectangle._center

            x, y = center.x, center.y
            h, w = shape.h, shape.w

            x_max = max(x_max, x + w/2)
            x_min = min(x_min, x - w/2)
            y_max = max(y_max, y + h/2)
            y_min = min(y_min, y - h/2)

        assert x_max > x_min and y_max > y_min
        return (x_max - x_min, y_max - y_min)

    G = nx.Graph()

    id2name = list[str](['']) # maps id => name, with a placeholder at id=0
    name2id = dict[str, int]()
    positions = dict[int, np.ndarray]()

    # add the modules as vertices
    for (id, module) in enumerate(netlist._modules, start=1):
        name: str = module.name
        fixed: bool = module.is_fixed
        area: float = module.area()
        center: Point | None = module.center
        width, height = bounding_box(module)

        if center:
            positions[id] = np.array([center.x, center.y])

        G.add_node(id, name=name, area=area, height=height, 
                   width=width, fixed=fixed)
        
        id2name.append(name)
        name2id[name] = id

    # add the hyperedges
    next_dummy_id: int = 1_000_000 # large number to avoid collision with module ids

    for net in netlist._edges:
        # represent the net as a set of module ids
        module_ids: set[int] = \
            [name2id[x.name] for x in net.modules]
        
        n_mod: int = len(module_ids)
        weight: float = net.weight

        if n_mod == 2: # normal edge
            G.add_edge(module_ids[0], module_ids[1], weight=weight)

        elif n_mod > 2: # hyperedge
            # calculate the center of mass of the modules in the net
            module_centers: list[np.ndarray | None] = \
                [positions.get(x) for x in module_ids]
            
            center_mass: np.ndarray | None = None

            if all(c is not None for c in module_centers):
                center_mass= np.mean(module_centers, axis=0)

            # add a dummy vertex
            G.add_node(next_dummy_id, center=center_mass, area=0,
                       dummy=True)
            
            # connect every module in the hyperedge to the dummy
            for module in module_ids:
                G.add_edge(next_dummy_id, module, weight=weight)

            next_dummy_id += 1
    
    return G, positions

def graph_to_netlist(netlist: Netlist, G: nx.Graph, 
                     positions: dict[int, np.ndarray]) -> Netlist:
    """
    Updates the module positions after the graph drawing algorithm.
    """

    for module_id, pos in positions.items():
        if G.nodes[module_id].get('dummy', False):
            continue

        new_center: Point = Point(tuple(pos))

        # get Module object
        module_name: str = G.nodes[module_id]['name']
        module: Module = netlist._name2module[module_name]

        if module.is_fixed():
            continue

        # update the center of the module and the center of all its rectangles
        old_center: Point = module.center
        shift: Point = new_center - old_center

        for rectangle in module.rectangles:
            rectangle.center += shift

        module.center = new_center

        # assign a square to modules with no shape 
        # (this requires having a center, as well as an area)
        if module.num_rectangles() == 0:
            module.create_square()

    return netlist

def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog,
                                     description="Relocate the modules of the netlist using a force-directed algorithm "
                                                 "to obtain better initial values for the next stages.")
    parser.add_argument("--netlist", required=True,
                        help="input netlist filename")
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME", required=True,
                        help="size of the die (width x height) or name of the file")
    parser.add_argument("--output", required=True,
                        help="output netlist file")
    
    parser.add_argument("--max_iter", default=100, type=int,
                        help="maximum number of ForceAtlas2 iterations")
    
    parser.add_argument("--seed", default=None, type=int,
                        help="random seed to generate positions for modules that don't have one")
    
    parser.add_argument("--scaling_factor", default=None, type=float,
                        help="Scaling factor between ForceAtlas2 layout size and die size. " + \
                             "If provided, it skips the scale estimation phase and speeds up layout generation")

    parser.add_argument("--scaling_factor_read", required=False, default=None,
                        help="File containing scaling factor between ForceAtlas2 layout size and die size. " + \
                             "If provided, it skips the scale estimation phase and speeds up layout generation")
    
    parser.add_argument("--scaling_factor_write", required=False, default=None,
                        help="file to write the scaling factor to, to speed up future runs")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    options: dict[str, str] = parse_options(prog, args)

    netlist_path: str = options['netlist']
    die_path: str = options['die']
    out_path: str = options['output']
    out_file_type: str = out_path[out_path.rfind('.'):]
    max_iter: int = options['max_iter']
    seed: float | None = options['seed']
    scaling_factor: float | None = options['scaling_factor']
    scaling_factor_read: str = options['scaling_factor_read']
    scaling_factor_write: str = options['scaling_factor_write']

    if out_file_type not in ['.json', '.yaml']:
        raise ValueError(f"Invalid output file type {out_path}, must be json or yaml")
    
    if scaling_factor is None and scaling_factor_read is not None:
        with open(scaling_factor_read, 'r') as f:
            try:
                scaling_factor = float(f.read())
            except ValueError:
                raise ValueError(f"Invalid {scaling_factor_read} file format")
   

    netlist: Netlist = Netlist(netlist_path)
    die : Die = Die(die_path)

    netlist_graph: nx.Graph = nx.Graph()
    positions: dict[int, np.ndarray] = {}
    netlist_graph, positions = netlist_to_graph(netlist)

    positions: dict[int, np.ndarray] = {}
    positions, scaling_factor = \
        forceatlas2_layout(G=netlist_graph, H=die.height, 
                           W=die.width, max_iter=max_iter,
                           pos=positions, scaling_factor=scaling_factor,
                           seed=seed)
    
    netlist =  graph_to_netlist(netlist, netlist_graph, positions)

    if out_file_type == '.json':
        netlist.write_json(out_path)

    elif out_file_type == '.yaml':
        netlist.write_yaml(out_path)

    if scaling_factor_write:
        with open(scaling_factor_write, 'w') as f:
            f.write(str(scaling_factor))

if __name__ == '__main__':
    main()