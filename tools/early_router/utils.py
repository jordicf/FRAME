from tools.early_router.types import EdgeID,NodeId
from collections import deque
import math


def to_scientific_notation(number):
    x = math.floor(math.log10(abs(number)))
    v = number / 10**x
    return v, x


def rescale(value:float, old_min:float=0, old_max=1, new_min=500, new_max=1500):
    assert value >= 0, "Formula do not hold for negative values"
    if abs(old_max - old_min) < 1e-6:
        print("Division by 0!")
        return (new_max - new_min)/2.
    new_value = 500 + (value - old_min)*(new_max - new_min)/(old_max - old_min)
    return new_value


def compute_node_degrees(route:list[dict[EdgeID, float]])->dict[NodeId,int]:
    node_degrees: dict[NodeId,int] = {}
    for edge_dict in route:
        # Each edge is a tuple: (node1, node2) where node is a tuple (int, int, int)
        for edge, weight in edge_dict.items():
            node1, node2 = edge
            node_degrees[node1] = node_degrees.get(node1, 0) + 1
            node_degrees[node2] = node_degrees.get(node2, 0) + 1
    return node_degrees


class UnionFind:
    """Union-Find (Disjoint Set) with path compression and union by rank."""
    def __init__(self, nodes):
        """Initialize Union-Find with nodes (arbitrary tuple IDs)."""
        self.parent = {node: node for node in nodes}  # Each node is its own parent
        self.rank = {node: 0 for node in nodes}  # Initialize ranks

    def find(self, u):
        """Find root of u with path compression."""
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        """Union by rank of two sets containing u and v."""
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True  # Edge added to MST
        return False  # Edge forms a cycle, ignore


def kruskal(nodes:set[NodeId], edges:list)->list[EdgeID]:
    """
    Kruskal's Algorithm for Minimum Spanning Tree (MST) with arbitrary node IDs.
    
    :param nodes: Set of node IDs (tuples like (x, y, z))
    :param edges: List of HananEdges3D representing edges
    :return: Minimum spanning tree cost and the edges in the MST
    """
    edges.sort(key=lambda e: e.length)  # Sort edges by weight (O(E log E))
    uf = UnionFind(nodes)
    mst_cost = 0
    mst_edges = []

    for e in edges:
        u = e.source._id
        v = e.target._id
        if uf.union(u, v):  # O(α(V)) ~ almost O(1)
            mst_edges.append((u, v))

    return mst_edges


def bfs_find_order(mst_graph, start, required_nodes):
    """Perform BFS and return nodes in the order they are found."""
    visited = set()
    queue = deque([start])
    found_order = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node in required_nodes:
            found_order.append(node)
        for neighbor in mst_graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    return found_order

# # Example Usage
# nodes = {
#     (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)
# }
# edges = [
#     (1, (0, 0, 0), (1, 0, 0)),
#     (4, (0, 0, 0), (0, 1, 0)),
#     (3, (1, 0, 0), (0, 1, 0)),
#     (2, (1, 0, 0), (1, 1, 0)),
#     (5, (0, 1, 0), (1, 1, 0))
# ]
# required_nodes = [(1, 0, 0), (0, 0, 0), (1, 1, 0)]  # Example: Need to connect these nodes

# # Compute MST
# mst_graph = kruskal(nodes, edges)

# # Find order of required nodes in BFS traversal
# found_order = bfs_find_order(mst_graph, required_nodes[0], set(required_nodes))

# print("Order of required nodes found in traversal:", found_order)