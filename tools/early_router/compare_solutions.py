import re
from typing import Dict, Set, Tuple, FrozenSet

def parse_solution_file(path: str) -> Dict[int, Set[FrozenSet[Tuple[int, int, int]]]]:
    """
    Parse a routing solution file.

    Returns a dict mapping net id to a set of edges, where each edge is a frozenset of two node tuples.
    Each node tuple is (x, y, z).
    """
    solutions: Dict[int, Set[FrozenSet[Tuple[int, int, int]]]] = {}
    edge_pattern = re.compile(r"\(([-\d]+),\s*([-\d]+),\s*([-\d]+)\)-\(([-\d]+),\s*([-\d]+),\s*([-\d]+)\)")

    with open(path, 'r') as f:
        current_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('n '):
                # Start of a new net block: 'n id num_edges weight'
                parts = line.split()
                current_id = int(parts[1])
                solutions[current_id] = set()
            elif line == '!':
                current_id = None
            elif current_id is not None:
                match = edge_pattern.match(line)
                if match:
                    x1, y1, z1, x2, y2, z2 = map(int, match.groups())
                    node_a = (x1, y1, z1)
                    node_b = (x2, y2, z2)
                    # Use frozenset so that order does not matter
                    solutions[current_id].add(frozenset((node_a, node_b)))
                else:
                    raise ValueError(f"Invalid edge line: {line}")
    return solutions


def compare_solution_files(path1: str, path2: str) -> Dict[int, int]:
    """
    Compare two routing solution files.

    Returns a dict mapping each net id (present in either file) to the number of differing edges
    (size of the symmetric difference between their edge sets).
    """
    sol1 = parse_solution_file(path1)
    sol2 = parse_solution_file(path2)

    all_ids = set(sol1.keys()) | set(sol2.keys())
    differences: Dict[int, int] = {}

    for net_id in all_ids:
        edges1 = sol1.get(net_id, set())
        edges2 = sol2.get(net_id, set())
        diff = edges1.symmetric_difference(edges2)
        if diff:
            differences[net_id] = len(diff)

    return differences

# Example usage:
# diffs = compare_solution_files('solution_a.txt', 'solution_b.txt')
# print(diffs)  # e.g., {65: 2, 71: 0}  # only nets with differences are included
