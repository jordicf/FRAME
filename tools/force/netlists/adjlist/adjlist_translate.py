from frame.utils.keywords import KW_MODULES, KW_NETS, KW_AREA, KW_CENTER
from ruamel.yaml import YAML
import sys 
import random


def translate(input_file, output_file, seed: int | None = None):
    """
    Translates a graph in list format (plain text adjacency list)
    into a netlist.
    """

    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = filter(lambda x: len(x) > 0, lines)
        edges = [tuple(line.split()) for line in lines]
        nodes = set([edge[0] for edge in edges] + [edge[1] for edge in edges])

    if seed is not None:
        random.seed(seed)

    area_nodes = 1.0

    node_to_frame_node = {node : "M_" + node for node in nodes}

    frame_modules = {node_to_frame_node[node] : {
        KW_AREA: area_nodes,
        KW_CENTER: [0, 0] if seed is None else [random.gauss(0, 0.01), random.gauss(0, 0.01)]
    } for node in sorted(nodes)}
    frame_edges = [[node_to_frame_node[edge[0]], node_to_frame_node[edge[1]]] for edge in edges]
    data = {KW_MODULES: frame_modules, KW_NETS: frame_edges}

    yaml = YAML()
    yaml.default_flow_style = False
    with open(output_file, 'w') as stream:
        yaml.dump(data, stream)


def main():
    assert len(sys.argv) in [2, 3]

    input_file = sys.argv[1] # fitxer de tipus ".list"

    if len(sys.argv) == 2:
        last_dot = input_file.rfind('.')
        output_file = input_file[:last_dot] + ".yaml"
    else:
        output_file = sys.argv[2] # fitxer de tipus .yaml

    translate(input_file, output_file)


if __name__ == "__main__":
    main()
