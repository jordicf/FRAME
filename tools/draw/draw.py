#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:47:48 2021

@author: Jordi Cortadella

Module to draw a netlist in which the modules are represented by rectangles
"""

import math
from distinctipy import distinctipy
from PIL import Image, ImageDraw, ImageFont
from typing import NamedTuple, Any
from frame.geometry.geometry import Point, Shape, Rectangle
from frame.netlist.netlist import Netlist
from frame.netlist.netlist_types import HyperEdge
from frame.netlist.module import Module
from frame.die.die import Die
from argparse import ArgumentParser


# Tuple to represent the scaling factors for a drawing
class Scaling(NamedTuple):
    xscale: float   # x scaling factor
    yscale: float   # y scaling factor
    width: int      # width of the picture (in pixels), without frame
    height: int     # height of the picture (in pixles), without frame
    frame: int      # thickness of the frame around canvas (in pixels)


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
        height = original.h * width / original.w
    if width == 0:
        width = original.w * height / original.h

    return Scaling(width / original.w, height / original.h, round(width), round(height), frame)


def calculate_bbox(netlist: Netlist) -> Shape:
    """
    Calculates the bounding box of the netlist using the rectangles
    :param netlist: the netlist
    :return: the bounding box
    """
    xmax, ymax = 0, 0
    if netlist.num_rectangles > 0:
        xmax = max([r.bounding_box[1].x for r in netlist.rectangles])
        ymax = max([r.bounding_box[1].y for r in netlist.rectangles])
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
    draw = ImageDraw.Draw(im)
    return im, draw


def calculate_centers(e: HyperEdge) -> list[Point]:
    """
    Calculates a list of points to be connected from a hyperedge. The first point acts as the center of the star
    :param e: The hyperedge
    :return: the list of points (the first point is the center of the star)
    """
    list_points = [Point(0, 0)]  # The center (initially a fake point)
    sum_x, sum_y = 0, 0
    for m in e.modules:
        c = m.center if m.num_rectangles == 0 else m.calculate_center_from_rectangles()
        assert c is not None, f'Cannot calculate center for block {m.name}'
        sum_x += c.x
        sum_y += c.y
        list_points.append(c)
    npoints = len(list_points) - 1
    list_points[0] = Point(sum_x / npoints, sum_y / npoints)
    return list_points


def draw_circle(im: Image, m: Module, color, scaling: Scaling, fontsize: int) -> None:
    """Draws a circle for block b. The area of the circle corresponds to the area of the b"""
    radius = math.sqrt(m.area() / math.pi)
    ll = Point(m.center.x - radius, m.center.y - radius)
    ur = Point(m.center.x + radius, m.center.y + radius)
    draw_geometry(im, ll, ur, color, scaling, m.name, fontsize, True)


def draw_rectangle(im: Image, r: Rectangle, color, scaling: Scaling, fontsize: int) -> None:
    """Draws the rectangle r"""
    ll, ur = r.bounding_box
    draw_geometry(im, ll, ur, color, scaling, r.name, fontsize, False)


def draw_geometry(im: Image, ll: Point, ur: Point, color, scaling: Scaling,
                  name: str, fontsize: int, circle: bool) -> None:
    transp = Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(transp, "RGBA")
    ll = scale(ll, scaling)
    ur = scale(ur, scaling)
    # Notice that y-coordinates are swapped
    if circle:
        draw.ellipse((ll.x, ur.y, ur.x, ll.y), fill=color)
    else:
        draw.rectangle((ll.x, ur.y, ur.x, ll.y), fill=color)
    # Now the name
    font = ImageFont.truetype("arial.ttf", fontsize)
    ccolor = distinctipy.get_text_color((color[0] / 255, color[1] / 255, color[2] / 255), threshold=0.6)[0]
    ccolor = round(ccolor * 255)
    ccolor = (ccolor, ccolor, ccolor)
    txt_w, txt_h = draw.textsize(name, font=font)  # To center the text
    txt_x, txt_y = round((ll.x + ur.x - txt_w) / 2), round((ll.y + ur.y - txt_h) / 2)
    draw.text((txt_x, txt_y), name, fill=ccolor, font=font, align="center", anchor="ms", stroke_width=1,
              stroke_fill=ccolor)
    im.paste(Image.alpha_composite(im, transp))


def draw(options: dict[str, Any]) -> int:
    infile = options['netlist']
    netlist = Netlist(infile)
    fontsize = options['fontsize']

    # Assignment of a color to each block
    colors = distinctipy.get_colors(netlist.num_modules)
    colors = [(math.floor(r * 255), math.floor(g * 255), math.floor(b * 255), 128) for (r, g, b) in colors]
    module2color = {b: colors[i] for i, b in enumerate(netlist.modules)}

    # Canvas
    layout = options['layout']
    if layout is not None:
        d = Die(layout)
        die = Shape(d.width, d.height)
    else:
        # Calculate bounding box
        die = calculate_bbox(netlist)

    # Scaling factors
    frame = options['frame']
    scaling = calculate_scaling(die, options['width'], options['height'], frame)
    im, draw = create_canvas(scaling)

    # Outer frame
    draw.rectangle((0, 0, scaling.width + 2 * frame, scaling.height + 2 * frame), outline=COLOR_GREY, width=4)
    # Inner frame
    draw.rectangle((frame, frame, scaling.width + frame, scaling.height + frame), outline=COLOR_WHITE, width=2)

    # Draw rectangles or circles of the blocks
    for m in netlist.modules:
        color = module2color[m]
        if m.num_rectangles == 0:
            draw_circle(im, m, color, scaling, fontsize)
        else:
            for r in m.rectangles:
                draw_rectangle(im, r, color, scaling, fontsize)

    # Draw edges
    for e in netlist.edges:
        list_points: list[Point] = calculate_centers(e)
        canvas_points = [scale(p, scaling) for p in list_points]
        center = canvas_points[0]
        for pin in canvas_points[1:]:
            draw.line([center, pin], fill=COLOR_WHITE, width=3)
        if len(canvas_points) > 3:  # Circle in the center
            rad = 8
            draw.ellipse([center.x - rad, center.y - rad, center.x + rad, center.y + rad], outline=COLOR_WHITE, width=3)

    # Output file
    outfile = options['outfile']
    if outfile is None:
        outfile = gen_out_filename(infile)
    im.save(outfile, quality=95)
    return 0


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool.

    :param prog: tool name.
    :param args: command-line arguments.
    :return: a dictionary with the arguments.
    """
    parser = ArgumentParser(prog=prog, description="A floorplan drawing tool", usage='%(prog)s [options]')
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("--die", help="Size of the die (width x height)", metavar="<width>x<height>")
    parser.add_argument("-o", "--outfile", help="output file (gif)")
    parser.add_argument("--width", type=int, default=0, help="width of the picture (in pixels)")
    parser.add_argument("--height", type=int, default=0, help="height of the picture (in pixels)")
    parser.add_argument("--frame", type=int, default=40, help="frame around the bounding box (in pixels)")
    parser.add_argument("--fontsize", type=int, default=20, help="text font size")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """Main function."""
    options = parse_options(prog, args)
    return draw(options)

if __name__ == "__main__":
    main()