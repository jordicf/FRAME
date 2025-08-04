# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Keywords for JSON/YAML files and dictionary keys
"""


class KW:
    """Class to store the keywords used in JSON/YAML files"""

    MODULES = "Modules"  # Modules of the netlist
    NETS = "Nets"  # Edges (hyperedges) of the netlist
    AREA = "area"  # Area (of a module, or rectangle)
    CENTER = "center"  # Center (of a module or rectangle)
    WIDTH = "width"  # Width of a rectangle
    HEIGHT = "height"  # Height of a rectangle
    SHAPE = "shape"  # A pair of width and height
    LL = "ll"  # LL corner of a rectangle
    UR = "ur"  # UR corner of a rectangle
    ASPECT_RATIO = "aspect_ratio"  # Aspect ratio of a module/rectangle
    TERMINAL = "terminal"  # Is the module a terminal?
    HARD = "hard"  # Hard module
    FIXED = "fixed"  # Is a module (or rectangle) fixed?
    FLIP = "flip"  # Can a non-rectangular hard module be flipped?
    RECTANGLES = "rectangles"  # For lists of rectangles
    REGION = "region"  # Region of a rectangle (e.g. LUT, BRAM, DSP)
    REGIONS = "regions"  # List of regions in the die

    # Name of the top (default) region for the die
    # Regions are used to define different slices in the die (e.g. LUT, DSP, BRAM, etc).
    # Regions are mostly used for FPGAs
    GROUND = "_"

    # Name for a blocked region
    BLOCKAGE = "#"  # String to represent a blocked region
