# (c) Antoni Pech Alberich 2024
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

from typing import Any, List, Tuple
from frame.utils.keywords import KW_MODULES, KW_NETS, KW_AREA, KW_FIXED, \
    KW_CENTER, KW_RECTANGLES, KW_TERMINAL, KW_HARD, KW_WIDTH, KW_HEIGHT
import argparse
import math
import numpy as np
from frame.geometry.geometry import Shape
from frame.netlist.module import Module
from frame.netlist.netlist_types import NamedHyperEdge
from tools.floorset_parser.strop import Strop
from frame.utils.utils import write_yaml
from frame.netlist.yaml_read_netlist import parse_yaml_module
from frame.netlist.yaml_write_netlist import dump_yaml_modules, dump_yaml_namededges

Point = Tuple[float, float]
Polygon = List[Point]
Rectangle = List[float]


class Floorplan():
    """
    A class to represent a floorplan.

    Attributes:
        _fp_data (dict[np.ndarray]): The raw floorplan data stored in a dictionary with keys: 
            ['area_blocks', 'b2b_connectivity', 'p2b_connectivity', 'pins_pos', 'placement_constraints',
            'vertex_blocks', 'b_tree', 'metrics']
        _d (float): A density percentage of the floorplan between 0 and 1.
        _modules (list[Module]): A list of parsed module objects, each representing a block or pin 
            with associated geometric and functional properties.
        _nets (list[NamedHyperEdge]): A list of parsed connectivity data, represented as hyperedges 
            between modules or pins, with associated weights.
        _width (float): The computed width of the die based on the floorplan data.
        _height (float): The computed height of the die based on the floorplan data.
        _alpha (float): A scaling parameter used for normalizing weights.

    Methods:
        _parse_modules():
            Parses and initializes module data from the floorplan dataset.
        _parse_connections(fp_data: dict[list]) -> list[NamedHyperEdge]:
            Parses and initializes connectivity data, computing hyperedges for
            block-to-block and pin-to-block connections.
        write_yaml:
            Writes the data into a yaml format.
    """
    _fp_data: dict[np.ndarray]
    "Floorplan raw data"
    _d: float
    "Density Percentage"
    _modules: list[Module]
    "List of modules (blocks and terminals)"
    _nets: list[NamedHyperEdge]
    "List of edge connections"
    _width: float
    "Die width"
    _height: float
    "Die height"
    _alpha: float
    "Scaling Factor"

    def __init__(self, floorplan_data: dict[np.ndarray], density: float) -> None:
        """For more information about the structure of floorplan_data check README.md"""

        keys = ['area_blocks', 'b2b_connectivity', 'p2b_connectivity',\
                 'pins_pos', 'placement_constraints', 'vertex_blocks', 'b_tree', 'metrics']
        
        assert isinstance(floorplan_data, dict), "Error floorplan data type. Has to be a dict."
        for k in floorplan_data.keys():
            assert k in keys, \
                f"Unknown key {k} in floorplan data.\nKeys allowed: {keys}"

        assert isinstance(floorplan_data['area_blocks'], np.ndarray), \
            "Wrong data type for area_blocks"
        assert not (floorplan_data['area_blocks'] < 0).any(), \
            "Negative values not allowed in area input"

        assert isinstance(floorplan_data['b2b_connectivity'], np.ndarray), \
            "Wrong data type for b2b_connectivity"
        assert not (floorplan_data['b2b_connectivity'] < 0).any(), \
            "Negative values not allowed in block-to-block connections"

        assert isinstance(floorplan_data['p2b_connectivity'], np.ndarray), \
            "Wrong data type for p2b_connectivity"
        assert not (floorplan_data['p2b_connectivity'] < 0).any(), \
            "Negative values not allowed in pin-to-block connections"
        
        assert isinstance(floorplan_data['pins_pos'], np.ndarray), \
            "Wrong data type for pins_pos"        
        assert not (floorplan_data['pins_pos'] < 0).any(), \
            "Negative values not allowed in pin positions"
        
        assert isinstance(floorplan_data['placement_constraints'], np.ndarray), \
            "Wrong data type for placement constraints"    
        assert not (floorplan_data['placement_constraints'] < 0).any(), \
            "Negative values not allowed in placement constraints"

        assert isinstance(floorplan_data['vertex_blocks'], np.ndarray), \
            "Wrong data type for vertex_blocks"

        assert isinstance(floorplan_data['metrics'], np.ndarray), \
            "Wrong data type for metrics"         
        assert not (floorplan_data['metrics'] < 0).any(), \
            "Negative values not allowed in any metric value"

        self._fp_data = floorplan_data
        self.num_modules = len(self._fp_data['area_blocks'])
        self.num_pins = self._fp_data['metrics'][1]
        self._modules = []
        self._nets = []
        
        assert isinstance(density, float) and 0 <= density <= 1, \
            "Wrong type for density factor, or value outof bounds [0,1]"
        self._d = float(density)
        
        self._parse_modules()
        self._parse_connections()

    def _parse_modules(self) -> None:
        """
        Parse and initialize module data from the floorplan dataset.

        This method processes block and pin data, computes the necessary properties, 
        and stores them in the `self._modules` list. It also validates the die area 
        against the width and height metrics from the floorplan.

        """
        for mod_id in range(self.num_modules):
            name = f"M{mod_id}"
            data = dict()
            
            vertices = self._fp_data['vertex_blocks'][mod_id]
            vertices = vertices[vertices[:, 0] != -1]
            if len(vertices) > 1 : # Handling Prime FloorSet
                data[KW_RECTANGLES] = strop_decomposition(vertices)
                cx, cy = compute_centroid(data[KW_RECTANGLES])
            elif len(vertices) == 1 : # Handling Lite FloorSet
                # In FloorSet a rectangle is stored as [w, h, x, y], where x,y
                # is the low-left point. In FRAME rectangles are stored as [cx,cy,w,h],
                # where c is the center.
                cx = float((vertices[2] + vertices[0]) / 2)
                cy = float((vertices[3] + vertices[1]) / 2)
                data[KW_RECTANGLES] = [cx, cy, float(vertices[0]), float(vertices[1])]
            else:
                pass # This should never happen

            # FloorSet   | FRAME constraint
            # Pre-placed | fixed
            # Fixed      | hard
            # Hard (and consequently fixed) cannot have area, center and must
            # have at least one rectangle
            if self._fp_data['placement_constraints'][mod_id][1]:
                data[KW_FIXED] = True
            elif self._fp_data['placement_constraints'][mod_id][0]:
                data[KW_HARD] = True
            else:
                data[KW_AREA] = float(self._fp_data['area_blocks'][mod_id])
                data[KW_CENTER] = [cx, cy]
            
            m = parse_yaml_module(name, data)
            self._modules.append(m)
        
        # For the die of the current floorplan
        shape_y = 0
        shape_x = 0
        for _id, pin_pos in enumerate(self._fp_data['pins_pos']):
            name = f"T{_id}"
            data = dict()
            data[KW_CENTER] = [pin_pos[0], pin_pos[1]]
            data[KW_TERMINAL] = True

            m = parse_yaml_module(name, data)
            self._modules.append(m)

            shape_x = max(shape_x, pin_pos[0])
            shape_y = max(shape_y, pin_pos[1])

        assert shape_x*shape_y == self._fp_data['area_blocks'].sum(), \
            "The die area do not match with the Width and Height"
        self._width = float(shape_y)
        self._height = float(shape_x)
    
    def _parse_connections(self) -> None:
        """
        Parse and initialize connectivity data for blocks and pins.

        This method processes block-to-block (b2b) and pin-to-block (p2b) connectivity, 
        computes the weight normalization factor (alpha), and creates hyperedges for 
        each connection storing them in `self._nets` list.
        """
        max_w = -1
        for mod_id in range(max(self.num_modules, int(self.num_pins))):
            bl_w = weight_sum(self._fp_data['b2b_connectivity'], 
                              self._fp_data['p2b_connectivity'], mod_id)
            if max_w < bl_w:
                max_w = bl_w
                max_bl_id = mod_id

        assert max_w > 0., "Inconsistency: The maximum weight of all blocks is 0 or lower" 
        perimeter = compute_perimeter(self._fp_data['vertex_blocks'][max_bl_id])
        self._alpha = float(self._d * perimeter / max_w)

        for b2b_edge in self._fp_data['b2b_connectivity']:
            b1, b2, w = b2b_edge
            net = NamedHyperEdge(modules=[f"M{int(b1)}", f"M{int(b2)}"], weight= float(w*self._alpha))         
            self._nets.append(net)

        for p2b_edge in self._fp_data['p2b_connectivity']:
            pin, bl, w = p2b_edge
            net = NamedHyperEdge(modules=[f"T{int(pin)}", f"M{int(bl)}"], weight= float(w*self._alpha))
            self._nets.append(net)

    def write_yaml_FPEF(self, filename: str | None = None) -> (str | None):
        """Writes the data into a YAML file. If no file name is given, a string with the yaml contents is returned"""
        data = {
            KW_MODULES: dump_yaml_modules(self.modules),
            KW_NETS: dump_yaml_namededges(self.nets)
        }
        return write_yaml(data, filename)

    def write_yaml_DIEF(self, filename: str | None = None) -> (str | None):
        """Writes the data into a YAML file. If no file name is given, a string with the yaml contents is returned"""
        data = {
            KW_WIDTH: self._width,
            KW_HEIGHT: self._height
        }
        return write_yaml(data, filename)
    
    @property
    def shape(self) -> Shape:
        """Returns the shape of the floorplan"""
        return Shape(self._width, self._height)
    
    @property
    def modules(self) -> list[Module]:
        """Returns the list of modules"""
        return self._modules

    @property
    def nets(self) -> list[NamedHyperEdge]:
        """Returns the list of hyperedges"""
        return self._nets
    
    @property
    def scaling_factor(self) -> float:
        """Returns the weight scaling factor"""
        return self._alpha
    
    @property
    def density_percentage(self) -> float:
        """Returns the density percentage set in this floorplan"""
        return self._d


def weight_sum(connections_b2b: np.ndarray, connections_p2b: np.ndarray, target_id: int) -> float:
    """
    Calculate the weighted sum of connections associated with a target block.

    Args:
        connections_b2b (np.ndarray): A 2D array representing block-to-block connections.
            Each row contains [block_id_1, block_id_2, weight].
        connections_p2b (np.ndarray): A 2D array representing pin-to-block connections.
            Each row contains [pin_id, block_id, weight].
        target_id (int): The ID of the target block to calculate the weighted sum for.

    Returns:
        float: The total weight of block-to-block and block-related pin connections
    """
    # Identify rows in block-to-block connections that involve the target block
    mask = (connections_b2b[:, 0] == target_id) | (connections_b2b[:, 1] == target_id)
    result = np.sum(connections_b2b[mask, 2])
    
    # Identify rows in pin-to-block connections where the target is a pin or block
    mask_block = (connections_p2b[:, 1] == target_id)
    result += np.sum(connections_p2b[mask_block, 2])

    return float(result)


def compute_perimeter(vertices: Polygon) -> float:
    """
    Compute the perimeter of a polygon given its vertices.

    Args:
        vertices (Polygon): A list or sequence of 2D points (tuples) representing the polygon's vertices. 
            The vertices are assumed to be ordered consecutively.

    Returns:
        float: The perimeter of the polygon.
    """
    perimeter = 0

    # Iterate over consecutive vertices to calculate edge lengths
    for i in range(len(vertices) - 1):
        x1, y1 = vertices[i]
        x2, y2 = vertices[i + 1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        perimeter += distance

    return float(perimeter)


def strop_decomposition(vertices: Polygon) -> list[Rectangle]:
    """
    Decomposes a polygon into Single-Trunk-Rectilinear-Orthogonal-Polygons.

    Args:
        vertices: List of tuples [(x1, y1), (x2, y2), ...] representing the
            polygon's vertices.

    Returns:
        rectangles: List of rectangles represented as [x,y,w,h], the center
            (x,y), the width w, and the height h.
    """
    rectangles = []
    # Extract unique x and y coordinates and sort them
    x_coords = sorted(set(x for x, y in vertices))
    y_coords = sorted(set(y for x, y in vertices), reverse=True)

    rows = len(y_coords) - 1
    cols = len(x_coords) - 1

    # Check each rectangle defined by consecutive x and y intervals
    m = ""
    for i in range(rows):
        for j in range(cols):
            # Define the rectangle bounds
            x_min, x_max = x_coords[j], x_coords[j + 1]
            y_max, y_min = y_coords[i], y_coords[i + 1]
            # Determine the center of the rectangle
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            # Check if the center is inside the polygon
            if is_point_inside_polygon((center_x, center_y), vertices):
                m += '1'
            else:
                m += '0'
        m += '\n'

    s = Strop(m)
    assert s.is_strop, f"Polygon is not a STROP {vertices}"
    sol = next(s.instances(), None)
    for r in sol.rectangles():
        x_min, x_max = x_coords[r.columns.low], x_coords[r.columns.high + 1]
        y_max, y_min = y_coords[r.rows.low], y_coords[r.rows.high + 1]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        rectangles.append([float(cx), float(cy), float(w), float(h)])

    return rectangles


def is_point_inside_polygon(point: Point, vertices: Polygon) -> bool:
    """
    Determine if a point is inside a polygon using the even-odd rule algorithm.

    Args:
        point: the point tuple (x,y) to check.
        vertices: a list of points representing the vertices of the polygon.
    
    Returns:
        A boolean whether the point is inside or not.
    """
    x, y = point
    n = len(vertices)
    inside = False

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        if (y1 <= y < y2) or (y2 <= y < y1):
            intersect_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersect_x:
                inside = not inside

    return inside


def compute_centroid(partition: List[Rectangle]) -> Point:
    """
    Compute the centroid of a simple polygon.

    Args:
        vertices: List of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]
                representing the polygon's vertices. The polygon should be
                closed (first vertex = last vertex).

    Returns:
        A tuple (Cx, Cy) representing the centroid of the polygon.
    """
    assert len(partition) > 0, "The partition should be not empty."
    n = len(partition)
    cx = 0
    cy = 0
    total_area = 0
    for i in range(n):
        x, y, w, h = partition[i]
        cx += w*h*x
        cy += w*h*y
        total_area += w*h
    cx /= (total_area)
    cy /= (total_area)
    return (float(cx), float(cy))


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

    # Input file argument
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the .npz or .npy file containing one input floorplan data."
    )
    parser.add_argument(
        "-d",
        "--connection_density",
        type=float,
        default=0.5,
        help="Percentage for connection density (default: 0.5).",
    )
    parser.add_argument(
        "--output_DIEF",
        type=str,
        help="Destination of the Die Exchange YAML output file.",
    )
    parser.add_argument(
        "--output_FPEF",
        type=str,
        help="Destination of the Floorplan Exchange YAML output file.",
    )

    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    infilepath: str = options['input']
    outfilepath_DIEF = options['output_DIEF']
    outfilepath_FPEF = options['output_FPEF']
    density = options['connection_density']

    assert infilepath.endswith(".npz"), "Invalid file type. Please provide a .npz or .npy file."
    floorplan_data = np.load(infilepath, allow_pickle=True)
    fp = Floorplan(dict(floorplan_data), density)

    if outfilepath_FPEF:
        fp.write_yaml_FPEF(outfilepath_FPEF)
    if outfilepath_DIEF:
        fp.write_yaml_DIEF(outfilepath_DIEF)
    return


if __name__ == "__main__":
    main()

