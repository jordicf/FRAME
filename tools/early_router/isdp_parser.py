#!/usr/bin/env python3
"""
ISDP-to-FPEF Netlist Converter

This script parses an ISDP benchmark file (global routing contest format) and converts it 
to an FPEF netlist format (YAML). In this conversion, each net's pin is represented as a 
separate module with a fixed location (its pin coordinates), and each net is translated into 
a hyperedge connecting those modules. The minimum routed width for the net is appended as 
the net weight (if greater than the default 1).

The ISDP file format is expected to have the following sections:

    grid # # # 
    vertical capacity # # # # #
    horizontal capacity # # # # #
    minimum width # # # # #
    minimum spacing # # # # #
    via spacing # # # # #
    lower_left_x lower_left_y tile_width tile_height

    num net #
    netname id_# number_of_pins minimum_width
    x y layer
    x y layer
    ...
    [repeat for each net]

    # capacity adjustments (optional)
    column row layer column row layer reduced_capacity_level
    [repeat]

Usage:
    python isdp_to_fpef.py input.isdp output_fpef.yaml

Dependencies:
    PyYAML (install via pip install pyyaml)
"""

import sys
import os
#print("\n".join(sys.path))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import re
import math
import yaml
from tools.early_router.hanan import HananCell, Layer
from frame.geometry.geometry import Point
from frame.netlist.netlist_types import NamedHyperEdge


def parse_isdp_file(filename):
    """
    Parses the ISDP file and returns a dictionary containing:
      - grid info (ignored for FPEF conversion)
      - nets: a list of nets; each net is a dict with:
          "netname": string
          "id": string
          "num_pins": int
          "min_width": float
          "pins": a list of dicts, each with keys "x", "y", "layer"
      - capacity_adjustments: (if any; ignored in conversion)
    """
    with open(filename, 'r') as f:
        # remove blank lines and comment lines (comments start with a character such as #)
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]

    i = 0
    # Parse the header lines (grid and capacities)
    header_keys = ["grid", "vertical capacity", "horizontal capacity", "minimum width",
                   "minimum spacing", "via spacing"]
    header = {}
    for key in header_keys:
        tokens = lines[i].split()
        if not " ".join(tokens[:len(key.split())]).lower() == key:
            print(f"Expected header line starting with '{key}' at line {i+1}")
            continue
        # Store the remaining tokens as numbers (convert to float/int as needed)
        # For simplicity, we store them as strings/numbers but we won't use them further.
        header[key] = tokens[len(key.split()):]
        i += 1

    # Parse the lower left and tile size line
    tokens = lines[i].split()
    if len(tokens) < 4:
        print("Expected lower_left_x, lower_left_y, tile_width, tile_height")
        grid_origin = None
    else:
        lower_left_x = float(tokens[0])
        lower_left_y = float(tokens[1])
        tile_width = float(tokens[2])
        tile_height = float(tokens[3])
        grid_origin = {"lower_left_x": lower_left_x, "lower_left_y": lower_left_y,
                    "tile_width": tile_width, "tile_height": tile_height}
        i += 1

    # Parse the number of nets
    tokens = lines[i].split()
    if tokens[0].lower() != "num" or tokens[1].lower() != "net":
        raise ValueError("Expected 'num net' line")
    num_nets = int(tokens[2])
    i += 1

    nets = []
    for net_idx in range(num_nets):
        # Parse net header: netname id_# number_of_pins minimum_width
        header_tokens = lines[i].split()
        if len(header_tokens) < 3:
            raise ValueError(f"Invalid net header at net {net_idx+1}")
        elif len(header_tokens) < 4:
            net = {
                "netname": header_tokens[0],
                "id": header_tokens[1],
                "num_pins": int(header_tokens[2]),
                "min_width": 1,
                "pins": []
            }
        else:
            net = {
                "netname": header_tokens[0],
                "id": header_tokens[1],
                "num_pins": int(header_tokens[2]),
                "min_width": float(header_tokens[3]),
                "pins": []
            }
        i += 1
        for _ in range(net["num_pins"]):
            pin_tokens = lines[i].split()
            if len(pin_tokens) < 2:
                raise ValueError(f"Invalid pin specification at net {net['netname']}")
            elif len(pin_tokens) < 3:
                pin = {
                    "x": float(pin_tokens[0]),
                    "y": float(pin_tokens[1])
                }
            else:
                # Note: The pin coordinates are given in absolute units.
                # They can be converted to tile indices if needed:
                #    tile_x = math.floor((pin_x - lower_left_x) / tile_width)
                #    tile_y = math.floor((pin_y - lower_left_y) / tile_height)
                pin = {
                    "x": float(pin_tokens[0]),
                    "y": float(pin_tokens[1]),
                    "layer": int(pin_tokens[2])
                }
            net["pins"].append(pin)
            i += 1
        nets.append(net)

    capacity_adjustments = None
    if i < len(lines):
        # Number of capacity adjustments
        tokens = lines[i].split()
        if len(tokens) > 1:
            raise ValueError(f"No information of how many adjustments are needed in line {i}")
        n_ajustments = int(tokens[0])
        i += 1
        capacity_adjustments = []
        while i < len(lines):
            tokens = lines[i].split()
            if len(tokens) < 7:
                raise ValueError("Expected 7 tokens in capacity adjustment line")
            adjustment = {
                "column": int(tokens[0]),
                "row": int(tokens[1]),
                "layer": int(tokens[2]),
                "target_column": int(tokens[3]),
                "target_row": int(tokens[4]),
                "target_layer": int(tokens[5]),
                "reduced_capacity": int(tokens[6])
            }
            capacity_adjustments.append(adjustment)
            i += 1

    return {
        "header": header,
        "grid_origin": grid_origin,
        "nets": nets,
        "capacity_adjustments": capacity_adjustments
    }


def convert_to_hanangrid(isdp_data):
    """
    Converts the parsed ISDP data into list of HananCells format.
    In this conversion, each pin is mapped to a module.
    The module name is built as: net_<netname>_<pin_index>.
    The module's center is set to the pin's (x, y) coordinate.
    The net is converted into a hyperedge connecting the module names.
    The net's min_width is appended as the hyperedge weight if not 1.
    """
    cells = []
    named_nets = dict()

    if isdp_data['grid_origin']:
        w=isdp_data['grid_origin']['tile_width'] + isdp_data['grid_origin']['lower_left_x']
        h=isdp_data['grid_origin']['tile_height'] + isdp_data['grid_origin']['lower_left_y']
    else:
        w=1
        h=1
    v_cap = isdp_data['header']['vertical capacity']
    h_cap = isdp_data['header']['horizontal capacity']
    if len(isdp_data['header']['grid'])>2:
        n_layers = int(isdp_data['header']['grid'][2])
        layers =[]
        for l in range(n_layers):
            d = 'H' if int(h_cap[l])>0 else 'V'
            layers.append(Layer(d, h_cap=int(h_cap[l]), v_cap=int(v_cap[l])))
    else:
        layers = [Layer('HV')]
    for i in range(int(isdp_data['header']['grid'][0])):
        for j in range(int(isdp_data['header']['grid'][1])):
            mod_name = f"M{i}_{j}"
            x = i*w + w/2
            y = j*h + h/2
            cells.append(HananCell((i,j),Point(x,y),h_cap[0],v_cap[0],mod_name))
    for net in isdp_data["nets"]:
        net_module_names = set()
        if net["num_pins"] > 1000:
            continue
        for idx, pin in enumerate(net["pins"]):
            mod_name = f"{net['netname']}_{idx+1}"
            # For each pin, create a module with its center.
            #    tile_x = math.floor((pin_x - lower_left_x) / tile_width)
            #    tile_y = math.floor((pin_y - lower_left_y) / tile_height)
            if isdp_data['grid_origin']:
                i = math.floor((pin["x"] - isdp_data['grid_origin']['lower_left_x']) / isdp_data['grid_origin']['tile_width'])
                j = math.floor((pin["y"] - isdp_data['grid_origin']['lower_left_y']) / isdp_data['grid_origin']['tile_height'])
            else:
                i = pin['x']
                j = pin['y']
            mod_name = f"M{i}_{j}"
            net_module_names.add(mod_name)
        if len(net_module_names) < 2:
            continue
        # In FPEF, a hyperedge is a list of module names.
        # Append the net's min_width as the last element if it is not the default value 1.
        if net["min_width"] != 1:
            net_entry = NamedHyperEdge(list(net_module_names), net["min_width"])
        else:
            net_entry = NamedHyperEdge(list(net_module_names), 1)
        named_nets[net["id"]]= net_entry

    return {
        "HananCells": cells,
        "Nets": named_nets,
        "Layers": layers
    }


def convert_to_fpef(isdp_data):
    """
    Converts the parsed ISDP data into FPEF netlist format.
    In this conversion, each pin is mapped to a module.
    The module name is built as: net_<netname>_<pin_index>.
    The module's center is set to the pin's (x, y) coordinate.
    The net is converted into a hyperedge connecting the module names.
    The net's min_width is appended as the hyperedge weight if not 1.
    """
    modules = {}
    fpef_nets = []

    if isdp_data['grid_origin']:
        w=isdp_data['grid_origin']['tile_width'] + isdp_data['grid_origin']['lower_left_x']
        h=isdp_data['grid_origin']['tile_height'] + isdp_data['grid_origin']['lower_left_y']
    else:
        w=1
        h=1
    for i in range(int(isdp_data['header']['grid'][0])):
        for j in range(int(isdp_data['header']['grid'][1])):
            mod_name = f"M{i}_{j}"
            x = i*w + w/2
            y = j*h + h/2
            modules[mod_name] = {
                "center": [x, y],
                "area": w*h,
                "rectangles": [[x,y,w,h]]
            }   
    for net in isdp_data["nets"]:
        net_module_names = set()
        if net["num_pins"] > 1000:
            continue
        for idx, pin in enumerate(net["pins"]):
            mod_name = f"{net['netname']}_{idx+1}"
            # For each pin, create a module with its center.
            #    tile_x = math.floor((pin_x - lower_left_x) / tile_width)
            #    tile_y = math.floor((pin_y - lower_left_y) / tile_height)
            if isdp_data['grid_origin']:
                i = math.floor((pin["x"] - isdp_data['grid_origin']['lower_left_x']) / isdp_data['grid_origin']['tile_width'])
                j = math.floor((pin["y"] - isdp_data['grid_origin']['lower_left_y']) / isdp_data['grid_origin']['tile_height'])
            else:
                i = pin['x']
                j = pin['y']
            mod_name = f"M{i}_{j}"
            net_module_names.add(mod_name)
        # In FPEF, a hyperedge is a list of module names.
        # Append the net's min_width as the last element if it is not the default value 1.
        if net["min_width"] != 1:
            net_entry = list(net_module_names) + [net["min_width"]]
        else:
            net_entry = list(net_module_names)
        if len(net_module_names) < 2:
            continue
        fpef_nets.append(net_entry)

    return {
        "Modules": modules,
        "Nets": fpef_nets
    }

