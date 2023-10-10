"""
Generate benchmark with random graphs for force algorithms comparison.
"""

import os
import sys
from tools.force.netlists.adjlist.generate_benchmark import generate
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point, NPoint
from tools.draw.draw import get_graph_plot
import networkx as nx
import numpy as np


SPRING_ITERATIONS = 300
SPRING_SEED = 73
log_file = ""


def log(*args, **kwargs):
    print(*args, **kwargs)
    with open(log_file, "a") as file:
        file.write(' '.join(args))
        file.write('\n')


def netlist_to_nx(netlist: Netlist):
    """
    Returns a networkx graph given a netlist.
    """

    G = nx.Graph()
    for module in netlist.modules:
        G.add_node(module.name)
    for edge in netlist.edges:
        G.add_weighted_edges_from([(edge.modules[0].name, edge.modules[1].name, edge.weight)])    
    return G


def force(file: str, heuristic: str, dimensions: int | None = None) -> str:
    """
    Returns the output netlist of the force algorithm given 
    an input netlist and a heuristic.
    """

    out_file = f"{file}-out-{heuristic}" + (str(dimensions) if dimensions is not None else "")
    if heuristic == "multi":
        if dimensions is None:
            dimensions = 3
        os.system(f"frame force --netlist {file}.yaml --out-netlist {out_file}.yaml --heuristic 1 -it {SPRING_ITERATIONS} --die 1x1 -n {dimensions} -p")
    elif heuristic == "flexrepel":
        os.system(f"frame force --netlist {file}.yaml --out-netlist {out_file}.yaml --heuristic 2 -it {SPRING_ITERATIONS} --die 1x1")
    return out_file


def multi2(file: str):
    """
    Returns the output netlist of the force algorithm given 
    an input netlist using the multidimensional heuristic
    with 2 dimensions.
    """

    return force(file, "multi", 2)

def multi3(file: str):
    """
    Returns the output netlist of the force algorithm given 
    an input netlist using the multidimensional heuristic
    with 3 dimensions.
    """

    return force(file, "multi", 3)

def multi4(file: str):
    """
    Returns the output netlist of the force algorithm given 
    an input netlist using the multidimensional heuristic
    with 4 dimensions.
    """

    return force(file, "multi", 4)

def multi8(file: str):
    """
    Returns the output netlist of the force algorithm given 
    an input netlist using the multidimensional heuristic
    with 8 dimensions.
    """

    return force(file, "multi", 8)


def spring(file: str ) -> str:
    """
    Returns the output netlist of the networkx spring
    layout given an input netlist.
    """

    out_file = f"{file}-out-spring"
    netlist = Netlist(file+".yaml")
    G = netlist_to_nx(netlist)
    pos = {}
    for module in netlist.modules:
        pos[module.name] = np.array([module.center.x, module.center.y])
    pos = nx.spring_layout(G, pos=pos, iterations=SPRING_ITERATIONS, seed=SPRING_SEED)
    for module in netlist.modules:
        module.center = Point(float(pos[module.name][0]), float(pos[module.name][1]))
    netlist.write_yaml(out_file+".yaml")
    return out_file


def kk(file: str) -> str:
    """
    Returns the output netlist of the networkx force
    (kamada kawai) algorithm given an input netlist.
    """

    out_file = f"{file}-out-kk"
    netlist = Netlist(file+".yaml")
    G = netlist_to_nx(netlist)
    pos = {}
    for module in netlist.modules:
        pos[module.name] = np.array([module.center.x, module.center.y])
    pos = nx.kamada_kawai_layout(G, pos=pos)
    for module in netlist.modules:
        module.center = Point(float(pos[module.name][0]), float(pos[module.name][1]))
    netlist.write_yaml(out_file+".yaml")
    return out_file


def kk_spring(file: str) -> str:
    return spring(kk(file))


def kk_force(file: str, heuristic: str) -> str:
    return force(kk(file), heuristic)


def kk_force_spring(file: str, heuristic: str):
    return spring(kk_force(file, heuristic))


def force_spring(file: str, heuristic: str):
    return spring(force(file, heuristic))


def multi_spring(file: str):
    return force_spring(file, "multi")


def flexrepel_spring(file: str):
    return force_spring(file, "flexrepel")


def kk_multi_spring(file: str):
    return kk_force_spring(file, "multi")


def kk_flexrepel_spring(file: str):
    return kk_force_spring(file, "flexrepel")


def multi(file: str):
    return force(file, "multi")


def flexrepel(file: str):
    return force(file, "flexrepel")


def normalized_wire_length_metric(file: str):
    """
    Returns the normalized wirelength metric given an
    input netlist.
    """

    netlist = Netlist(file+".yaml")
    x_min = min(module.center.x for module in netlist.modules)
    x_max = max(module.center.x for module in netlist.modules)
    y_min = min(module.center.y for module in netlist.modules)
    y_max = max(module.center.y for module in netlist.modules)
    assert (x_max > x_min + 1e-9) and (y_max > y_min + 1e-9)
    for module in netlist.modules:
        module.center.x = (module.center.x - x_min) / (x_max - x_min)
        module.center.y = (module.center.y - y_min) / (y_max - y_min)
    return netlist.wire_length


def wire_length_metric(file: str):
    netlist = Netlist(file+".yaml")
    return netlist.wire_length


metrics = [
    normalized_wire_length_metric,
    wire_length_metric,
]

alternatives = [
    multi2,
    multi3,
    multi4,
    multi8,
    multi_spring,
    flexrepel_spring,
    flexrepel,
    spring,
    kk_multi_spring,
    kk_flexrepel_spring,
    kk_spring,
]


def main():
    global log_file 

    assert len(sys.argv) == 2
    seed = sys.argv[1]
    log_file = f"benchmark_{seed}.log"

    log("STARTING BENCHMARK")

    directory = f"benchmark_{seed}"

    os.system(f"rm -rf {directory}")

    files = generate(seed, directory)

    for file in files:
        log("graph: ", file)
        for alternative in alternatives:
            log("   >>", alternative.__name__.ljust(15))
            out_file = alternative(file)
            for metric in metrics:
                metric_val = metric(out_file)
                log(" "*8, metric.__name__.ljust(32), str(metric_val).ljust(16))

    log("BENCHMARK FINISHED!!!")

if __name__ == '__main__':
    main()
