# (c) Antoni Pech Alberich 2024
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import numpy as np
from tools.floorset_parser.floor_set_manager.utils.utils import compute_centroid, compute_perimeter, weight_sum, \
    strop_decomposition
from frame.netlist.yaml_write_netlist import dump_yaml_namededges
from frame.utils.utils import write_json_yaml
from frame.utils.keywords import KW

from frame.geometry.geometry import Point  # check if it is the same
from frame.netlist.netlist_types import NamedHyperEdge
from typing import Any


EPSILON = 1e-3


class FloorSetInstance():
    """
    A class to represent a floorset instance.
    """
    _fp_data: dict[str,np.ndarray]
    "Floorplan raw data"
    _d: float | None
    "Density Percentage"
    _modules: dict
    "List of modules (blocks and terminals)"
    _nets: list[NamedHyperEdge]
    "List of edge connections"
    _width: float
    "Die width"
    _height: float
    "Die height"

    def __init__(self, floorplan_data: dict[str,np.ndarray], density: float | None, terminals_as_modules: bool) -> None:
        """
        :Args:
        :param dict[np.ndarray] floorplan_data: The raw floorplan data stored in a dictionary with keys: 
            ['area_blocks', 'b2b_connectivity', 'p2b_connectivity', 'pins_pos', 'placement_constraints',
            'vertex_blocks', 'b_tree', 'metrics']
        :param float density: A density percentage of the floorplan between 0 and 1. If None, no rescaling is made.
        :param bool terminals_as_modules: whether store terminals pins as modules with rectangle of size 10^-3.
        """

        keys = ['area_blocks', 'b2b_connectivity', 'p2b_connectivity',
                'pins_pos', 'placement_constraints', 'vertex_blocks', 'b_tree', 'metrics']

        assert isinstance(
            floorplan_data, dict), "Error floorplan data type. Has to be a dict."
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
        self._modules = dict()
        self._nets = []

        if density:
            assert isinstance(density, float) and 0 <= density <= 1, \
                "Wrong type for density factor, or value outof bounds [0,1]"
            self._d = float(density)
        else:
            self._d = density
        self._alpha:float|None = None

        self._parse_modules(terminals_as_modules)
        self._parse_connections()

    def _parse_modules(self, terminals_as_modules: bool) -> None:
        """
        Parse and initialize module data from the floorplan dataset.

        This method processes block and pin data, computes the necessary properties, 
        and stores them in the `self._modules` list.

        """
        for mod_id in range(self.num_modules):
            name = f"M{mod_id}"
            data: dict[str,Any] = dict()

            vertices = self._fp_data['vertex_blocks'][mod_id]
            vertices = vertices[vertices[:, 0] != -1]
            if len(vertices) > 1:  # Handling Prime FloorSet
                data[KW.RECTANGLES] = strop_decomposition(vertices)
                c = compute_centroid(data[KW.RECTANGLES])
            elif len(vertices) == 1:  # Handling Lite FloorSet
                # In FloorSet a rectangle is stored as [w, h, x, y], where x,y
                # is the low-left point. In FRAME rectangles are stored as [cx,cy,w,h],
                # where c is the center.
                cx = float((vertices[2] + vertices[0]) / 2)
                cy = float((vertices[3] + vertices[1]) / 2)
                data[KW.RECTANGLES] = [cx, cy, float(
                    vertices[0]), float(vertices[1])]
                c = Point(cx, cy)
            else:
                pass  # This should never happen

            # FloorSet   | FRAME constraint
            # Pre-placed | fixed
            # Fixed      | hard
            # Hard (and consequently fixed) cannot have area, center and must
            # have at least one rectangle
            if self._fp_data['placement_constraints'][mod_id][1]:
                data[KW.FIXED] = True
            elif self._fp_data['placement_constraints'][mod_id][0]:
                data[KW.HARD] = True
            else:
                data[KW.AREA] = float(self._fp_data['area_blocks'][mod_id])
                data[KW.CENTER] = [c.x, c.y]

            self._modules[name] = data

        # For the die of the current floorplan
        shape_x = max(p[0] for p in self._fp_data['pins_pos'])
        shape_y = max(p[1] for p in self._fp_data['pins_pos'])
        for _id, pin_pos in enumerate(self._fp_data['pins_pos']):
            name = f"T{_id}"
            data = dict()
            if terminals_as_modules:
                h = EPSILON
                w = EPSILON
                if float(pin_pos[0]) < EPSILON:
                    x = float(pin_pos[0]) + EPSILON
                elif float(pin_pos[0]) >= shape_x + EPSILON:
                    x = float(pin_pos[0]) - EPSILON
                if float(pin_pos[1]) < EPSILON:
                    y = float(pin_pos[1]) + EPSILON
                elif float(pin_pos[1]) >= shape_y + EPSILON:
                    y = float(pin_pos[1]) - EPSILON
                data[KW.RECTANGLES] = [x, y, w, h]
                data[KW.FIXED] = True
            else:
                data[KW.CENTER] = [float(pin_pos[0]), float(pin_pos[1])]
                data[KW.TERMINAL] = True

            self._modules[name] = data

        self._width = float(shape_x)
        self._height = float(shape_y)

    def _parse_connections(self) -> None:
        """
        Parse and initialize connectivity data for blocks and pins.

        This method processes block-to-block (b2b) and pin-to-block (p2b) connectivity, 
        computes the weight normalization factor (alpha), and creates hyperedges for 
        each connection storing them in `self._nets` list.
        """
        if self._d:
            max_f = -1.
            for mod_id in range(self.num_modules):
                bl_w = weight_sum(self._fp_data['b2b_connectivity'],
                                  self._fp_data['p2b_connectivity'], mod_id)
                vertices = self._fp_data['vertex_blocks'][mod_id]
                vertices = vertices[vertices[:, 0] != -1]
                perimeter = compute_perimeter(vertices)
                f = bl_w/perimeter
                if max_f < f:
                    max_f = f
            self._alpha = float(self._d / max_f)
        elif not self._alpha:
            self._alpha = 1

        for b2b_edge in self._fp_data['b2b_connectivity']:
            b1, b2, w = b2b_edge
            wei = float(w*self._alpha)
            net = NamedHyperEdge(
                modules=[f"M{int(b1)}", f"M{int(b2)}"], weight=wei if wei > 0 else 1)
            self._nets.append(net)

        for p2b_edge in self._fp_data['p2b_connectivity']:
            pin, bl, w = p2b_edge
            wei = float(w*self._alpha)
            net = NamedHyperEdge(
                modules=[f"T{int(pin)}", f"M{int(bl)}"], weight=wei if wei > 0 else 1)
            self._nets.append(net)

    def write_yaml_FPEF(self, filename: str | None = None) -> (str | None):
        """Writes the data into a YAML file. If no file name is given, a string with the yaml contents is returned"""
        data = {
            KW.MODULES: self.modules,
            KW.NETS: dump_yaml_namededges(self.nets)
        }
        return write_json_yaml(data, False, filename)

    def write_yaml_DIEF(self, filename: str | None = None) -> (str | None):
        """Writes the data into a YAML file. If no file name is given, a string with the yaml contents is returned"""
        data = {
            KW.WIDTH: self._width,
            KW.HEIGHT: self._height
        }
        return write_json_yaml(data, False, filename)

    @property
    def shape(self) -> tuple:
        """Returns the shape of the floorplan"""
        return (self._width, self._height)

    @property
    def modules(self) -> dict:
        """Returns the list of modules"""
        return self._modules

    @property
    def nets(self) -> list[NamedHyperEdge]:
        """Returns the list of nameshyperedges"""
        return self._nets

    @property
    def density_percentage(self) -> float | None:
        """Returns the density percentage set in this floorplan"""
        return self._d
