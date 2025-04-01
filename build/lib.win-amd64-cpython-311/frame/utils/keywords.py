# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Keywords for YAML files and dictionary keys
"""

KW_MODULES = "Modules"  # Modules of the netlist
KW_NETS = "Nets"  # Edges (hyperedges) of the netlist
KW_AREA = "area"  # Area (of a module, or rectangle)
KW_CENTER = "center"  # Center (of a module or rectangle)
KW_WIDTH = "width"  # Width of a rectangle
KW_HEIGHT = "height"  # Height of a rectangle
KW_SHAPE = "shape"  # A pair of width and height
KW_ASPECT_RATIO = "aspect_ratio"  # Aspect ratio of a module/rectangle
KW_TERMINAL = "terminal"  # Is the module a terminal?
KW_HARD = "hard"  # Hard module
KW_FIXED = "fixed"  # Is a module (or rectangle) fixed?
KW_FLIP = "flip"  # Can a non-rectangular hard module be flipped?
KW_RECTANGLES = "rectangles"  # For lists of rectangles
KW_REGION = "region"  # Region of a rectangle (e.g. LUT, BRAM, DSP)
KW_REGIONS = "regions"  # List of regions in the die
KW_NAME = "name"  # Name of an object

# Name of the top (default) region for the die
# Regions are used to define different slices in the die (e.g. LUT, DSP, BRAM, etc).
# Regions are mostly used for FPGAs
KW_GROUND = "_"

# Name for a blocked region
KW_BLOCKAGE = "#"  # String to represent a blocked region
