# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""
Package to draw a netlist in which the modules are represented by rectangles
"""

import math
from argparse import ArgumentParser
from typing import Any
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
from distinctipy import distinctipy

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape, Rectangle, BoundingBox
from frame.netlist.module import Module
from frame.netlist.netlist import Netlist
from frame.netlist.netlist_types import HyperEdge
from frame.allocation.allocation import Allocation


# Tuple to represent the scaling factors for a drawing
@dataclass()
class Scaling:
    xscale: float  # x scaling factor
    yscale: float  # y scaling factor
    width: int  # width of the picture (in pixels), without frame
    height: int  # height of the picture (in pixels), without frame
    frame: int  # thickness of the frame around canvas (in pixels)


# Some colors
COLOR_WHITE = (255, 255, 255)
COLOR_GREY = (128, 128, 128)
COLOR_BLACK = (0, 0, 0)


def scale(p: Point, s: Scaling) -> Point:
    """
    Scales a point according to the scaling factors
    :param p: the point
    :param s: the scaling tuple
    :return: the new point (with integer values)
    """
    return Point(round(p.x * s.xscale) + s.frame, s.height + s.frame - round(p.y * s.yscale))


def calculate_scaling(original: Shape, width: int, height: int, frame: int = 20, default: int = 1000) -> Scaling:
    """
    Calculates the scaling factor for a picture. If both width and height are zero,
    then the scaling preserves the aspect ratio and the max dimension is 1000
    :param original: original width and height of the picture
    :param width: desired width (0 if aspect ratio must be preserved)
    :param height: desired height (0 if aspect ratio must be preserved)
    :param frame: size of the frame around the canvas (default = 20)
    :param default: default size if width and height are zero
    :return: the shape of the canvas (in pixels) and the scaling factors
    """
    if width != 0 and height != 0:
        # Scale independently without preserving aspect ratio
        return Scaling(width / original.w, height / original.h, round(width), round(height), frame)

    if width == 0 and height == 0:
        # Let us assign the default value to the maximum size
        if original.w > original.h:
            width = default
        else:
            height = default

    if height == 0:
        height = round(original.h * width / original.w)
    if width == 0:
        width = round(original.w * height / original.h)

    return Scaling(width / original.w, height / original.h, width, height, frame)


def check_modules(modules: list[Module]) -> None:
    """
    Check that all modules are drawable, i.e., we either have a list of rectangles or a center and an area
    :param modules: list of modules
    :raises exception: if some module is not drawable
    """
    for m in modules:
        if m.num_rectangles > 0:
            continue
        assert m.center is not None, f'module {m.name} is not drawable. It has neither center nor rectangles.'
        assert m.area() > 0 or m.is_terminal, f'module {m.name} is not drawable (no area specified).'


def calculate_bbox(netlist: Netlist) -> Shape:
    """
    Calculates the bounding box of the netlist using the rectangles. Only xmax and ymax are returned.
    The ll corner of the bounding box is assumed to be (0,0)
    :param netlist: the netlist
    :return: the bounding box
    """
    xmax, ymax = 0.0, 0.0
    if netlist.num_rectangles > 0:
        xmax = max([r.bounding_box.ur.x for r in netlist.rectangles])
        ymax = max([r.bounding_box.ur.y for r in netlist.rectangles])
    # Check now the centers of the blocks
    for m in netlist.modules:
        c = m.center
        if c is not None:
            radius = math.sqrt(m.area() / math.pi)
            xmax, ymax = max(xmax, c.x + radius), max(ymax, c.y + radius)
    return Shape(xmax, ymax)


def gen_out_filename(name: str, suffix: str = "gif") -> str:
    """
    Generates the output filename by substituting the extension by the suffix.
    If the name has no extension ('.' is not found), the suffix is added at the end of the name
    :param name: name of the input file
    :param suffix: suffix of the output file
    :return: the output filename
    """
    i = name.rfind('.')
    if i < 0:
        return name + '.' + suffix
    return name[:i + 1] + suffix


def create_canvas(s: Scaling):
    """
    Generates the canvas of the drawing
    :param s: scaling of the layout
    """
    im = Image.new('RGBA', (s.width + 2 * s.frame, s.height + 2 * s.frame), (0, 0, 0, 255))
    drawing = ImageDraw.Draw(im)
    return im, drawing


def calculate_centers(e: HyperEdge, alloc: Allocation | None) -> list[Point]:
    """
    Calculates a list of points to be connected from a hyperedge. The first point acts as the center of the star
    :param e: The hyperedge
    :param alloc: an allocation of modules to rectangles
    :return: the list of points (the first point is the center of the star)
    """
    list_points = [Point(0, 0)]  # The center (initially a fake point)
    sum_x, sum_y = 0.0, 0.0
    for m in e.modules:
        if alloc is not None:
            c = alloc.center(m.name)
        else:
            assert m.center is not None
            c = m.center if m.num_rectangles == 0 else m.calculate_center_from_rectangles()
        assert c is not None, f'Cannot calculate center for module {m.name}'
        sum_x += c.x
        sum_y += c.y
        list_points.append(c)
    npoints = len(list_points) - 1
    list_points[0] = Point(sum_x / npoints, sum_y / npoints)
    return list_points


def draw_circle(im: Image.Image, m: Module, color, scaling: Scaling, fontsize: int = 0) -> None:
    """Draws a circle for block b. The area of the circle corresponds to the area of the b"""
    radius = math.sqrt(m.area() / math.pi)
    assert m.center is not None
    bb = BoundingBox(ll=Point(m.center.x - radius, m.center.y - radius),
                     ur=Point(m.center.x + radius, m.center.y + radius))
    draw_geometry(im, bb, color, scaling, m.name, fontsize, True)


def draw_rectangle(im: Image.Image, r: Rectangle, name: str, color, scaling: Scaling, fontsize: int = 0) -> None:
    """Draws the rectangle r"""
    bb = r.bounding_box
    draw_geometry(im, bb, color, scaling, name, fontsize, False)


def draw_geometry(im: Image.Image, bb: BoundingBox, color, scaling: Scaling,
                  name: str, fontsize: int, circle: bool) -> None:
    transp = Image.new('RGBA', im.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(transp, "RGBA")
    ll = scale(bb.ll, scaling)
    ur = scale(bb.ur, scaling)
    # Notice that y-coordinates are swapped
    if circle:
        drawing.ellipse((ll.x, ur.y, ur.x, ll.y), fill=color)
    else:
        drawing.rectangle((ll.x, ur.y, ur.x, ll.y), fill=color)
    # Now the name
    if fontsize > 0:
        font = ImageFont.truetype("arial.ttf", fontsize)
        ccolor = distinctipy.get_text_color((color[0] / 255, color[1] / 255, color[2] / 255), threshold=0.6)[0]
        ccolor = round(ccolor * 255)
        ccolor = (ccolor, ccolor, ccolor)
        txt_w, txt_h = drawing.textsize(name, font=font)  # To center the text
        txt_x, txt_y = round((ll.x + ur.x - txt_w) / 2), round((ll.y + ur.y - txt_h) / 2)
        drawing.text((txt_x, txt_y), name, fill=ccolor, font=font, align="center",
                     anchor="ms", stroke_width=1, stroke_fill=ccolor)
    im.paste(Image.alpha_composite(im, transp))


def get_floorplan_plot(netlist: Netlist, die_shape: Shape, allocation: Allocation | None = None, width: int = 0,
                       height: int = 0, frame: int = 20, fontsize: int = 20) -> Image.Image:
    """
    Generates the plot of the floorplan
    :param netlist: the netlist with the modules and the hyperedges
    :param allocation: allocation of modules to rectangles
    :param die_shape: the shape of the die
    :param width: desired width (0 if aspect ratio must be preserved)
    :param height: desired height (0 if aspect ratio must be preserved)
    :param frame: size of the frame around the canvas
    :param fontsize: text font size
    :return: PIL Image containing the floorplan plot
    """
    # Check that all modules are drawable
    check_modules(netlist.modules)

    # Assignment of a color to each block
    colors = distinctipy.get_colors(netlist.num_modules, rng=0)
    colors = [(round(r * 255), round(g * 255), round(b * 255), 128) for (r, g, b) in colors]
    module2color = {b: colors[i] for i, b in enumerate(netlist.modules)}

    # Scaling factors
    scaling = calculate_scaling(die_shape, width, height, frame)

    im, drawing = create_canvas(scaling)

    # Outer frame
    drawing.rectangle((0, 0, scaling.width + 2 * frame, scaling.height + 2 * frame), outline=COLOR_GREY, width=4)
    # Inner frame
    drawing.rectangle((frame, frame, scaling.width + frame, scaling.height + frame), outline=COLOR_WHITE, width=2)

    if allocation is not None:
        for rect_alloc in allocation.allocations:
            r = rect_alloc.rect
            bb = r.bounding_box
            ll, ur = scale(bb.ll, scaling), scale(bb.ur, scaling)
            a = rect_alloc.alloc
            color = (0, 0, 0, 128)
            for module, ratio in a.items():
                mcolor = module2color[netlist.get_module(module)]
                color = (color[0] + mcolor[0] * ratio, color[1] + mcolor[1] * ratio, color[2] + mcolor[2] * ratio, 128)
            color = (round(color[0]), round(color[1]), round(color[2]), 128)
            drawing.rectangle((ll.x, ur.y, ur.x, ll.y), fill=color)
        # Draw the module names
        font = ImageFont.truetype("arial.ttf", fontsize)
        for m in netlist.modules:
            center = scale(allocation.center(m.name), scaling)
            txt_w, txt_h = drawing.textsize(m.name, font=font)  # To center the text
            txt_x, txt_y = center.x - txt_w / 2, center.y - txt_h / 2
            drawing.text((txt_x, txt_y), m.name, fill=COLOR_WHITE, font=font, align="center",
                         anchor="ms", stroke_width=1, stroke_fill=COLOR_WHITE)
    else:
        # If no allocation, draw rectangles or circles of the modules
        for m in netlist.modules:
            color = module2color[m]
            if m.num_rectangles == 0:
                draw_circle(im, m, color, scaling, fontsize)
            else:
                name = m.name
                for i, r in enumerate(m.rectangles):
                    rname = name if m.num_rectangles == 1 else f"{name}[{i}]"
                    draw_rectangle(im, r, rname, color, scaling, fontsize)

    # Draw edges
    for e in netlist.edges:
        list_points: list[Point] = calculate_centers(e, allocation)
        canvas_points = [scale(p, scaling) for p in list_points]
        center = canvas_points[0]
        for pin in canvas_points[1:]:
            drawing.line([(center.x, center.y), (pin.x, pin.y)], fill=COLOR_WHITE, width=3)
        if len(canvas_points) > 3:  # Circle in the center
            rad = 8
            drawing.ellipse([center.x - rad, center.y - rad, center.x + rad, center.y + rad],
                            outline=COLOR_WHITE, width=3)

    return im


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="A floorplan drawing tool.", usage='%(prog)s [options]')
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("-d", "--die", metavar="<WIDTH>x<HEIGHT> or FILENAME",
                        help="size of the die (width x height) or name of the file")
    parser.add_argument("--alloc", metavar="FILENAME",
                        help="allocation file of modules to rectangles", )
    parser.add_argument("-o", "--outfile", help="output file (gif)")
    parser.add_argument("--width", type=int, default=0, help="width of the picture (in pixels)")
    parser.add_argument("--height", type=int, default=0, help="height of the picture (in pixels)")
    parser.add_argument("--frame", type=int, default=40, help="frame around the die (in pixels). Default: 40")
    parser.add_argument("--fontsize", type=int, default=20, help="text font size. Default: 20")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    infile = options['netlist']
    netlist = Netlist(infile)

    # Rectangle allocation
    alloc_option = options['alloc']
    allocation: Allocation | None = None
    if alloc_option is not None:
        allocation = Allocation(alloc_option)

    # Canvas
    alloc_die = Shape(0, 0)
    if allocation is not None:
        bb = allocation.bounding_box.bounding_box
        alloc_die = Shape(bb.ur.x, bb.ur.y)

    die_file = options['die']
    if die_file is not None:
        die = Die(die_file)
        die_shape = Shape(die.width, die.height)
    elif alloc_option is not None:
        die_shape = alloc_die
    else:
        die_shape = calculate_bbox(netlist)

    assert alloc_die.w <= die_shape.w and alloc_die.h <= die_shape.h

    im = get_floorplan_plot(netlist, die_shape, allocation, options['width'], options['height'], options['frame'],
                            options['fontsize'])

    # Output file
    outfile = options['outfile']
    if outfile is None:
        outfile = gen_out_filename(infile)
    im.save(outfile, quality=95)


if __name__ == "__main__":
    main()
