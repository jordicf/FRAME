import argparse
import networkx as nx
import numpy as np
from .contraction import contract
from .expansion import expand

from frame.die.die import Die
from frame.netlist.netlist import Netlist
from frame.netlist.netlist_types import Module
from frame.geometry.geometry import Point, Shape, Rectangle

from typing import Any

def _cos_similarity(u: np.ndarray, v: np.ndarray) -> float:
    assert u.size == v.size

    u = u.flatten()
    v = v.flatten()

    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))

def _get_heights_widths(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    heights = list[float]()
    widths =  list[float]()

    for v in G.nodes():
        if G.nodes[v].get('dummy', False):
            continue

        heights.append(G.nodes[v]['height'])
        widths.append(G.nodes[v]['width'])

    return np.array(heights), np.array(widths)

def _get_areas(G: nx.Graph) -> np.ndarray:
    areas = list[float]()

    for v in G.nodes():
        if G.nodes[v].get('dummy', False):
            continue

        areas.append(G.nodes[v]['area'])
    
    return np.array(areas)

def _get_centers(G: nx.Graph) -> np.ndarray:
    centers = list[np.ndarray]()

    for v in G.nodes():
        if G.nodes[v].get('dummy', False):
            continue

        centers.append(G.nodes[v]['center'])
    
    return np.array(centers)

def _get_fixed(G: nx.Graph) -> set[int]:
    fixed = set[int]()
    for v in G.nodes:
        if G.nodes[v].get('fixed', False):
            fixed.add(v)

    return fixed

def _run_atrrep(G: nx.Graph, heights: np.ndarray, widths: np.ndarray,
                areas: np.ndarray, centers: np.ndarray, H: float, W: float,
                hyperparams: dict, fixed: set[int] = {}) -> None:
    
    """
    Runs the whole repulsion - attraction algorithm

    Params:
        G: netlist in graph format
    """
    
    epsilon = hyperparams.get('epsilon', 1e-7)
    iter = 1

    ar_min = np.array([G.nodes[v].get('ar_min', 0) for v in range(len(centers))])
    ar_max = np.array([G.nodes[v].get('ar_max', float('inf')) for v in range(len(centers))])

    while True:
        print(f"Starting big iteration {iter}")
        
        old_centers: np.ndarray = centers.copy()
        contract(G, H, W, fixed, hyperparams)
        centers = _get_centers(G)

        expand(G, centers, heights, widths, areas, ar_min, ar_max, 
               H, W, fixed, hyperparams)
        
        centers = _get_centers(G)
        heights, widths = _get_heights_widths(G)
        
        iter += 1

        # exit condition
        print(_cos_similarity(centers, old_centers))
        if _cos_similarity(centers, old_centers) >= 1 - epsilon:
            break


def netlist_to_graph(netlist: Netlist) -> nx.Graph:
    """
    Converts the netlist into a networkx Graph, which is the
    required object type for the contraction algorithm (for now).

    Since hyperedges are not supported in networkx, a dummy
    vertex is added for each of them.
    """

    def width_height(module: Module) -> tuple[float, float]:
        """
        Returns the width & height of a rectangular module.
        If the shape of the module hasn't been defined yet,
        it assumes it is a square.
        It is assumed that the module has a defined center.
        """

        # if shape is undefined, assign a square 
        # (note: this requires having a center)
        if module.num_rectangles == 0:
            module.create_square()
        
        if module.num_rectangles > 1:
            raise NotImplementedError(
                "Repatr does not support non-rectangular modules yet"
            )
        
        # 1 rectangle only
        shape: Shape = module._rectangles[0].shape
        return (shape.h, shape.w)
    
    G = nx.Graph()

    id2name = list[str]() # maps id => name
    name2id = dict[str, int]()

    # add the modules as vertices
    for (id, module) in enumerate(netlist._modules):
        name: str = module.name

        center: Point | None  = module.center
        assert center is not None, \
            "repatr tool cannot be used with non-initialized centers. Perhaps " + \
            "initialize them with 'grdraw' or 'force'?"

        area: float = module.area()
        terminal: bool = module.is_iopin
        fixed: bool = module.is_fixed

        width, height = width_height(module)

        G.add_node(id, name=name, center=np.array([center.x, center.y]), 
                   area=area, height=height, width=width,
                   terminal=terminal, fixed=fixed)
        
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
            module_centers: list[np.ndarray] = \
                [G.nodes[x]['center'] for x in module_ids]
            
            center_mass: np.ndarray = []

            if all(c is not None for c in module_centers):
                center_mass = np.mean(module_centers, axis=0)

            # add a dummy vertex
            G.add_node(next_dummy_id, center=center_mass, area=0,
                       dummy=True)
            
            # connect every module in the hyperedge to the dummy
            for module in module_ids:
                G.add_edge(next_dummy_id, module, weight=weight)

            next_dummy_id += 1
    return G

def graph_to_netlist(netlist: Netlist, G: nx.Graph) -> Netlist:
    """
    Updates the center positions after the graph drawing algorithm.
    """

    for module_id in G.nodes:
        if G.nodes[module_id].get('dummy', False):
            continue

        new_center: Point = Point(tuple(map(float, G.nodes[module_id]['center'])))
        h: float = float(G.nodes[module_id]['height'])
        w: float = float(G.nodes[module_id]['width'])

        # get Module object
        module_name: str = G.nodes[module_id]['name']
        module: Module = netlist._name2module[module_name]

        # update the center
        module.center = new_center

        # update the shape
        module._rectangles = [Rectangle(center=new_center, shape=Shape(w=w, h=h))]

    return netlist

def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog,
                                     description="Relocate and reshape the modules of the netlist using "
                                                 "the attraction-repulsion algorithm.")
    parser.add_argument("--netlist", required=True,
                        help="input netlist filename")
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME", required=True,
                        help="size of the die (width x height) or name of the file")
    parser.add_argument("--output", required=True,
                        help="output netlist file")
    parser.add_argument("--stopping_tolerance", default=1e-4, type=float,
                        help="minimum relative change to stop the main loop (the higher the earlier stop)")
    parser.add_argument("--overlap_tolerance", default=1e-3, type=float,
                        help="tolerance to stop the expansion phase (the lower, the less overlap but more cpu time)")
    
    return vars(parser.parse_args(args))

def main(prog: str | None = None, args: list[str] | None = None) -> None:
    options: dict[str, str] = parse_options(prog, args)

    netlist_path: str = options['netlist']
    die_path: str = options['die']
    out_path: str = options['output']
    out_file_type: str = out_path[out_path.rfind('.'):]
    stopping_tolerance: float = options['stopping_tolerance']
    overlap_tolerance: float = options['overlap_tolerance']

    if out_file_type not in ['.json', '.yaml']:
        raise ValueError(f"Invalid output file type {out_path}, must be json or yaml")

    netlist: Netlist = Netlist(netlist_path)
    netlist_graph: nx.Graph = netlist_to_graph(netlist)

    die : Die = Die(die_path)

    H: float = die.height
    W: float = die.width

    hyperparams = {
        'epsilon': stopping_tolerance, # threshold of change to stop the main loop

        'contraction': {
            'niter': 5, # number of contractions
            'niter_decay': 1,
            'step_length': 2,
            'step_length_decay': 0.8,
            'step_length_min': 0
        },

        'expansion': {
            'epsilon': overlap_tolerance,# improvement to stop expanding
            'epsilon_decay': 0.8,
            'epsilon_min': 1e-4,
            
            'repel_rectangles' : {
                'epsilon': 1e-2, # improvement to stop iterating
                'epsion_decay': 0.9,
                'min_epsilon': 1e-4,
                'resolution': 1e-3, # minimum relative distance between hanan grid cells
                'resolution_decay': 1
            },

            'pseudo_solver': {
                'epsilon': 1e-5, # improvement to stop iterating
                'lr': 1, # initial learning rate
                'lr_decay': 0.9
            }
        }
    }

    heights, widths = _get_heights_widths(netlist_graph)
    areas = _get_areas(netlist_graph)
    centers = _get_centers(netlist_graph)
    fixed: set[int] = _get_fixed(netlist_graph)
    
    _run_atrrep(G=netlist_graph, heights=heights, widths=widths, areas=areas,
             centers=centers, H=H, W=W, hyperparams=hyperparams, 
             fixed=fixed)
    
    netlist = graph_to_netlist(netlist, netlist_graph)

    if out_file_type == '.json':
        netlist.write_json(out_path)

    elif out_file_type == '.yaml':
        netlist.write_yaml(out_path)

if __name__ == '__main__':
    main()
