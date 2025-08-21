import math
from frame.netlist.netlist import Netlist
from frame.netlist.netlist_types import HyperEdge, NamedHyperEdge
from frame.geometry.geometry import Point, Shape
from dataclasses import dataclass
from tools.early_router.types import NodeId, EdgeId
import itertools
from typing import cast


def manhattan_dist(p: Point, q: Point) -> float:
    return abs(p.x - q.x) + abs(p.y - q.y)


@dataclass
class HananCell:
    _id: tuple[int, int]  # Identifier
    center: Point  # Center
    width_capacity: float  # width capacity
    height_capacity: float  # height capacity
    modulename: str  # Module name


class HananGrid:
    """Construct a Hanan Grid from a Netlist without considering the terminals
    or a list of hanancells"""

    _cells: dict[tuple[int, int], HananCell]

    def __init__(self, netlist_or_cells: Netlist | list[HananCell]):
        if isinstance(netlist_or_cells, Netlist):
            # Collect all unique x-coordinates and y-coordinates
            x_coords = set()
            y_coords = set()

            rectangles = netlist_or_cells.rectangles  # Terminals are not here

            for r in rectangles:
                # Add the two x and y coordinates
                x_coords.add(r.bounding_box.ll.x)
                x_coords.add(r.bounding_box.ur.x)
                y_coords.add(r.bounding_box.ll.y)
                y_coords.add(r.bounding_box.ur.y)

            self._shape = Shape(
                w=float(max(x_coords) - min(x_coords)),
                h=float(max(y_coords) - min(y_coords)),
            )

            xcoords2pos = {x: i for i, x in enumerate(sorted(list(x_coords)))}
            xpos2coords = {i: x for i, x in enumerate(sorted(list(x_coords)))}
            ycoords2pos = {y: i for i, y in enumerate(sorted(list(y_coords)))}
            y2coords = {i: y for i, y in enumerate(sorted(list(y_coords)))}

            self._cells = {}
            # Create a dict with keys r'[i][j] and value a cell to an
            # identified rectangle
            for m in netlist_or_cells.modules:
                if m.is_iopin:
                    # No cells involving terminals
                    # (created in the graph as nodes)
                    continue

                for r in m.rectangles:
                    # For each rectangle get which cells do they fill
                    ll = r.bounding_box.ll  # Lower-Left
                    ur = r.bounding_box.ur  # Upper-Rigth
                    minx, miny = xcoords2pos[ll.x], ycoords2pos[ll.y]
                    maxx, maxy = xcoords2pos[ur.x], ycoords2pos[ur.y]
                    # Fill each cell
                    for i in range(minx, maxx):
                        for j in range(miny, maxy):
                            w = xpos2coords[i + 1] - xpos2coords[i]
                            h = y2coords[j + 1] - y2coords[j]
                            x = xpos2coords[i] + w / 2
                            y = y2coords[j] + h / 2
                            cell = HananCell(
                                _id=(i, j),
                                center=Point((x, y)),
                                width_capacity=w,
                                height_capacity=h,
                                modulename=m.name,
                            )
                            self._cells[(i, j)] = cell
        else:
            self._shape = Shape(0, 0)
            self._cells = {cell._id: cell for cell in netlist_or_cells}

    @property
    def cells(self) -> list[HananCell]:
        return list(self._cells.values())

    @property
    def shape(self) -> Shape:
        return self._shape

    def get_adjacent_cells(self, cell: HananCell) -> list[HananCell | None]:
        """
        Get the adjacent cells from a HananCell.

        Args:
            cell (HananCell)

        Returns:
            list[HananCell]: A list containing the adjacent cells.
        """
        curr_id = cell._id
        nei = [
            (curr_id[0] - 1, curr_id[1]),
            (curr_id[0] + 1, curr_id[1]),
            (curr_id[0], curr_id[1] - 1),
            (curr_id[0], curr_id[1] + 1),
        ]
        adjacent_cells = []
        for n in nei:
            if n in self._cells.keys():
                adjacent_cells.append(self.get_cell(n))
        return adjacent_cells

    def get_cell(self, id: tuple[int, int]) -> HananCell | None:
        if id not in self._cells.keys():
            return None
        return self._cells[id]

    def get_closest_cell_to_point(self, p: Point) -> HananCell | None:
        """Returns the HananCell that is the closest to Point p.
        The distance is computed with Manhattan distance."""
        return_cell = None
        curr_min = math.inf
        # Find Manhattan distance
        # |x1 - x2| + |y1 - y2|
        for cell in self._cells.values():
            dist = manhattan_dist(cell.center, p)
            if dist < curr_min:
                return_cell = cell
                curr_min = dist
        return return_cell


class Layer:
    """Represents a metal layer in a routing process."""

    def __init__(
        self,
        direction: str,
        pitch: float | None = None,
        name: str = "",
        h_cap=None,
        v_cap=None,
    ):
        """
        Initialize a Layer instance.

        :param direction: Allowed values - 'H' (Horizontal), 'V' (Vertical), 'HV' (Both)
        :param pitch: Metal pitch for the layer (default is 1.0)
        :param name: Optional name of the layer (default is an empty string)
        :param layer_id: Optional unique identifier for the layer (default is 0)
        """
        if direction not in {"H", "V", "HV"}:
            raise ValueError("Invalid direction! Allowed values: 'H', 'V', 'HV'")

        self.direction: str = direction
        self.pitch: float | None = pitch
        self.name: str = name
        self.h_cap = h_cap
        self.v_cap = v_cap

    def __repr__(self) -> str:
        """String representation of the Layer instance."""
        return f"Layer(name='{self.name}', direction='{self.direction}', pitch={self.pitch}, h_cap={self.h_cap}, v_cap={self.v_cap})"


@dataclass
class HananNode3D:
    _id: NodeId
    center: Point
    modulename: str


@dataclass
class HananEdge3D:
    source: HananNode3D  # Cell from Edge leaves
    target: HananNode3D  # Cell to Edge goes
    length: float  # Edge Length
    capacity: float  # Edge capacity
    crossing: bool  # Whether a it is a change of block
    via: bool  # Whether it is a change of via


class HananGraph3D:
    """Hanan Graph extended from a Hanan Grid, adding edges to adjacent cells and terminals.
    Also, adds super nodes for each module"""

    _nodes: dict[NodeId, HananNode3D]
    _edges: list[HananEdge3D]
    _adj_list: dict[NodeId, dict[NodeId, HananEdge3D]]

    def __init__(
        self,
        hanan_grid: HananGrid,
        layers: list[Layer],
        netlist: Netlist | None = None,
        **kwargs,
    ):
        """
        **kwargs:
        asap7: whether to use the pitches in asap7 tech (assumes microns)
        """
        self._hanan_grid = hanan_grid
        self._nodes = {}
        self._edges = []
        self._adj_list = {}

        assert all([isinstance(l, Layer) for l in layers]), (
            "layers values are not instances of Layer class"
        )
        self._layers = {i: l for i, l in enumerate(layers)}
        asap7 = kwargs.get("asap7", False)

        # Each cell become a Hanan node
        for cell in hanan_grid.cells:
            # Duplicate the node for each layer
            for l_id in self._layers:
                node_id = (cell._id[0], cell._id[1], l_id)
                self._nodes[node_id] = HananNode3D(
                    _id=node_id, center=cell.center, modulename=cell.modulename
                )
                if l_id != 0:
                    node_below = (cell._id[0], cell._id[1], l_id - 1)
                    # Add the vias connections
                    self._adj_list.setdefault(node_id, {})[node_below] = (
                        self.add_edge3D(node_id, node_below)
                    )
                    self._adj_list.setdefault(node_below, {})[node_id] = (
                        self.add_edge3D(node_below, node_id)
                    )

        # Connect the nodes from the Hanan Grid
        for layer_id in self._layers:
            for cell in hanan_grid.cells:
                adj_cells = hanan_grid.get_adjacent_cells(cell)
                for adj_cell in adj_cells:
                    if adj_cell and cell._id[0] == adj_cell._id[0]:
                        # They move vertical
                        cap = cell.width_capacity  # Assuming microns
                        direction = "V"
                        if self.layers[layer_id].v_cap:
                            cap = self.layers[layer_id].v_cap
                    else:
                        # They move horizontal
                        cap = cell.height_capacity  # Assuming microns
                        direction = "H"
                        if self.layers[layer_id].h_cap:
                            cap = self.layers[layer_id].h_cap
                    if adj_cell and direction in self.layers[layer_id].direction:
                        source_id = (cell._id[0], cell._id[1], layer_id)
                        target_id = (adj_cell._id[0], adj_cell._id[1], layer_id)
                        if asap7 and cap:
                            # floor(4000/76) = 52 wires for 4 μm edge and 76nm pitch
                            p = self.layers[layer_id].pitch
                            if p:
                                new_cap = int(cap * 1000 / p)
                            else:
                                new_cap = int(cap * 1000)
                            self._adj_list.setdefault(source_id, {})[target_id] = (
                                self.add_edge3D(source_id, target_id, new_cap)
                            )
                        else:
                            self._adj_list.setdefault(source_id, {})[target_id] = (
                                self.add_edge3D(source_id, target_id, cap)
                            )
                        # No need to add target -> source, because we will visit target and it will be included
        if netlist:
            self._add_terminals(hanan_grid, netlist)

    def _add_terminals(self, hanan_grid: HananGrid, netlist: Netlist):
        # Adding Terminals
        t = 0
        for m in netlist.modules:
            if not m.center:
                continue
            if m.is_iopin:
                # Check module, Terminals always have a defined center
                # Terminals are always on the lowest layer
                terminal_id = (t, -1, 0)
                terminal = HananNode3D(
                    _id=terminal_id, center=m.center, modulename=m.name
                )
                self._nodes[terminal_id] = terminal
                t += 1
                # Get the module that terminal is closest to, and create an edge
                cell = hanan_grid.get_closest_cell_to_point(m.center)
                if not cell:
                    continue
                # Connect the terminal to all layers.
                # TODO For now just on the lowest level 0
                # node_id = (cell._id[0], cell._id[1], 0)
                # self._adj_list.setdefault(terminal_id, {})[node_id] = self.add_edge3D(terminal_id, node_id)
                # self._adj_list.setdefault(node_id, {})[terminal_id] = self.add_edge3D(node_id, terminal_id)
                for layer_id in self._layers:
                    node_id = (cell._id[0], cell._id[1], layer_id)
                    self._adj_list.setdefault(terminal_id, {})[node_id] = (
                        self.add_edge3D(terminal_id, node_id, terminal=True)
                    )
                    self._adj_list.setdefault(node_id, {})[terminal_id] = (
                        self.add_edge3D(node_id, terminal_id, terminal=True)
                    )

    def add_edge3D(
        self,
        source_id: NodeId,
        target_id: NodeId,
        capacity: float = math.inf,
        terminal: bool = False,
    ) -> HananEdge3D:
        source = self._nodes.get(source_id, None)
        target = self._nodes.get(target_id, None)
        assert source and target, (
            "source_id or target_id not created when trying to add an edge"
        )
        l = manhattan_dist(source.center, target.center)
        edge = HananEdge3D(
            source=source,
            target=target,
            length=l,
            capacity=capacity,
            crossing=source.modulename != target.modulename,
            via=False if terminal else source_id[2] != target_id[2],
        )
        self._edges.append(edge)
        return edge

    def get_nodes_by_modulename(self, module_name: str) -> list[HananNode3D]:
        """
        Args:
            module_name (str): Module name

        Returns:
            list[HananNode3D]: A list containing the nodes with the same module name.
        """
        return [node for node in self._nodes.values() if node.modulename == module_name]

    def get_crossings_by_modulename(
        self, module_name: str
    ) -> dict[str, list[HananEdge3D]]:
        """
        Given a module name, returns the incoming and outgoing edges that are crossings.

        Args:
            module_name (str): The name of the module.

        Returns:
            dict[str, list[HananEdge3D]]: A dictionary containing incoming and outgoing crossing edges.
        """
        nodes = self.get_nodes_by_modulename(module_name)

        return {
            "in": [
                e
                for n in nodes
                for nei in self._adj_list[n._id]
                if (e := self.get_edge(nei, n._id)) and e.crossing
            ],
            "out": [
                e
                for n in nodes
                for nei in self._adj_list[n._id]
                if (e := self.get_edge(n._id, nei)) and e.crossing
            ],
        }

    @property
    def edges(self) -> list[HananEdge3D]:
        return self._edges

    @property
    def nodes(self) -> list[HananNode3D]:
        return list(self._nodes.values())

    @property
    def nodes_ids(self) -> list[NodeId]:
        return list(self._nodes.keys())

    @property
    def adjacent_list(self):
        return self._adj_list

    @property
    def layers(self) -> dict[int, Layer]:
        return self._layers

    @property
    def hanan_grid(self) -> HananGrid:
        return self._hanan_grid

    def get_edges_from_node(self, node: HananNode3D) -> dict[str, list[HananEdge3D]]:
        """Given a HananNode returns a dict with incoming and outcoming edges to that node"""
        # self._adj_list[n1][n2] is the edge from n1-> n2
        return {
            "in": [self._adj_list[nei][node._id] for nei in self._adj_list[node._id]],
            "out": list(self._adj_list[node._id].values()),
        }

    def is_terminal(self, node: HananNode3D) -> bool:
        """Check if the node is a terminal"""
        return node._id[1] == -1

    def get_node(self, id: NodeId) -> HananNode3D | None:
        if id in self._nodes.keys():
            return self._nodes[id]
        return None

    def get_adj_nodes(self, node: HananNode3D) -> list[HananNode3D]:
        """Returns a list of adjacent nodes to the given one. The input is not included in the return"""
        node_id = node._id
        return [cast(HananNode3D, self.get_node(n)) for n in self._adj_list[node_id]]

    def get_edge(self, source_id: NodeId, target_id: NodeId) -> HananEdge3D | None:
        if (source_id in self._adj_list) and (target_id in self._adj_list[source_id]):
            return self._adj_list[source_id][target_id]
        return None

    def get_net_boundingbox(
        self, net: HyperEdge | NamedHyperEdge, augmentation: int = 0
    ) -> list[HananNode3D]:
        """
        Given a net and an augmentation, create an augmented bounding box.

        Args:
            net (HyperEdge | NamedHyperEdge): The net (can have multiple pins).
            augmentation (int): The amount to expand the bounding box.

        Returns:
            list[HananNode3D]: A list with nodes inside the augmented bounding box.
        """
        selected = []
        all_nodes = []

        for m in net.modules:
            modules = self.get_nodes_by_modulename(m if isinstance(m, str) else m.name)
            if any(self.is_terminal(n) for n in modules):
                terminal = modules[0]
                selected.append(terminal)
                # Assuming one adjacent node
                all_nodes.append([self.get_adj_nodes(terminal)[0]._id])
            else:
                # Only keep layer 0 for speed-up
                all_nodes.append([n._id for n in modules if n._id[2] == 0])

        # Now, compute the minimal bounding box across all choices of one node from each module.
        best_area = None
        best_bbox = None  # Will store tuple (min_x, max_x, min_y, max_y)

        # Iterate over every combination: one candidate from each module.
        # TODO  think an heuristic to speed-up the process
        for combo in itertools.product(*all_nodes):
            xs = [n_id[0] for n_id in combo]
            ys = [n_id[1] for n_id in combo]
            # Compute the bounding box from the chosen combination
            current_min_x = min(xs)
            current_max_x = max(xs)
            current_min_y = min(ys)
            current_max_y = max(ys)

            # Compute area
            area = (current_max_x - current_min_x) * (current_max_y - current_min_y)
            if best_area is None or area < best_area:
                best_area = area
                best_bbox = (
                    max(current_min_x - augmentation, 0),
                    current_max_x + augmentation,
                    max(current_min_y - augmentation, 0),
                    current_max_y + augmentation,
                )

        # If no bounding box was computed, return an empty list.
        if best_bbox is None:
            return []

        min_x, max_x, min_y, max_y = best_bbox

        # Now, select all nodes from the grid (self.nodes) that fall within this bounding box.
        bounding_box_nodes = [
            node
            for node in self.nodes
            if min_x <= node._id[0] <= max_x and min_y <= node._id[1] <= max_y
        ]

        selected.extend(bounding_box_nodes)
        return selected

    def get_edgesid_subset(self, nodes: list[HananNode3D]) -> set[EdgeId]:
        """
        Given a list of nodes, returns a list of edges where both endpoints are in the given node list.

        Args:
            nodes (list[HananNode3D]): List of nodes.

        Returns:
            list[HananEdge3D]: List of edges connecting only the given nodes.
        """
        node_set = {n._id for n in nodes}
        return set(
            (e.source._id, e.target._id)
            for n in nodes
            for a, e in self.adjacent_list[n._id].items()
            if a in node_set
        )

    def apply_capacity_adjustments(self, cap_adjust: dict[EdgeId, float | int]) -> None:
        for e_id in cap_adjust:
            e = self.get_edge(e_id[0], e_id[1])
            r_e = self.get_edge(e_id[1], e_id[0])
            if e:
                e.capacity = cap_adjust[e_id]
            if r_e:
                r_e.capacity = cap_adjust[e_id]
