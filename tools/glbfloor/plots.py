# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""This file contains the code for plotting the floor plans"""

from dataclasses import astuple, dataclass
from math import ceil

from PIL import Image, ImageDraw, ImageFont
from distinctipy import get_text_color
from matplotlib import cm

from frame.geometry.geometry import Point
from frame.die.die import Die
from frame.allocation.allocation import Allocation
from tools.draw.draw import get_floorplan_plot as get_joint_floorplan_plot


class PlottingOptions:
    """Plotting options"""

    def __init__(self, name: str | None = None,
                 joint_plot: bool = False, separated_plot: bool = False, visualize: bool = False):
        """
        Constructor
        :param name: name of the plot to be produced in each optimization. The optimization number, the plot type, and
        the file extension are added automatically
        :param joint_plot: if True, produce an image at each iteration showing a joint floorplan plot
        :param separated_plot: if True, produce an image at each iteration showing a separated floorplan plot
        :param visualize: if True, produce an animation to visualize the complete optimization process
        """
        self.name = name
        self.joint_plot = joint_plot
        self.separated_plot = separated_plot
        self.visualize = visualize
        self._check_options()

    def _check_options(self):
        assert not self.visualize or (self.joint_plot or self.separated_plot), \
            "joint_plot or separated_plot must be True if visualize is True"
        assert not self.joint_plot or self.name, "plot name is required if joint_plot is True"
        assert not self.separated_plot or self.name, "plot name is required if separated_plot is True"
        assert not self.visualize or self.name, "plot name is required if visualize is True"


@dataclass()
class Scaling:
    """Auxiliary class to scale points to image coordinates"""
    scale_factor: float
    x_offset: int
    y_offset: int
    grid_width: int
    grid_height: int

    def scale(self, p: Point) -> tuple[int, int]:
        """Scale a point to an image coordinate"""
        return (round(p.x * self.scale_factor + self.x_offset),
                round(self.grid_height - p.y * self.scale_factor + self.y_offset))


def get_color(ratio: float, color_map: str) -> tuple[int, int, int, int]:
    """
    Get the color associated to the ratio, using the indicated matplotlib color map
    :param ratio: number between 0 and 1
    :param color_map: matplotlib color map name. See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    :return: the color as an RGBA integer tuple (in the range [0, 255])
    """
    color = [int(float(component) * 255) for component in cm.get_cmap(color_map)(round(ratio * 255))]
    return color[0], color[1], color[2], color[3]


def get_separated_floorplan_plot(die: Die, allocation: Allocation, dispersions: dict[str, float] | None = None,
                                 alpha: float | None = None, suptitle: str = "",
                                 draw_borders: bool = True, draw_ratios: bool = True, draw_text: bool = False) \
        -> Image.Image:
    """
    Return a PIL Image containing a floorplan plot, given the netlist and allocation of each module in each cell, and
    additional information for annotations.

    The plot is made up of separated subplots for each module. Module info (area, centroid, and dispersion) and model
    and floorplan info (alpha, total dispersion, total wire length, and objective function value) can also be displayed
    if provided. An optional title can be shown too.
    """
    assert die.netlist is not None, "No netlist associated to the die"

    margin = 40
    scale_factor = 150
    font_name = "arial.ttf"
    cell_font = ImageFont.truetype(font_name, round(scale_factor / 10))
    medium_font = ImageFont.truetype(font_name, round(scale_factor / 8))
    large_font = ImageFont.truetype(font_name, round(scale_factor / 6))
    color_map = "Blues"

    grid_width, grid_height = map(ceil, astuple(die.bounding_box.shape))
    grid_width *= scale_factor
    grid_height *= scale_factor

    n_modules = die.netlist.num_modules

    img = Image.new("RGB", (margin + (grid_width + margin) * n_modules, grid_height + 2 * margin), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    s = Scaling(scale_factor, margin, margin, grid_width, grid_height)

    refinable, fixed = die.floorplanning_rectangles()

    for module in die.netlist.modules:
        for rect in refinable + fixed:
            unscaled_bbox = rect.bounding_box
            draw.rectangle((s.scale(unscaled_bbox.ll), s.scale(unscaled_bbox.ur)),
                           fill=get_color(0, color_map),
                           outline="Black" if draw_borders else None)

        for module_alloc in allocation.allocation_module(module.name):
            rect = allocation.allocation_rectangle(module_alloc.rect_index).rect
            unscaled_bbox = rect.bounding_box
            bbox_min, bbox_max = s.scale(unscaled_bbox.ll), s.scale(unscaled_bbox.ur)

            ratio = module_alloc.area_ratio
            color = get_color(ratio, color_map)

            # Draw cells
            draw.rectangle((bbox_min, bbox_max),
                           fill=color,
                           outline="Black" if draw_borders else None)

            # Draw cell annotations
            if draw_ratios:
                cell_width, cell_height = bbox_max[0] - bbox_min[0], bbox_min[1] - bbox_max[1]
                assert cell_width > 0 and cell_height > 0
                cell_text = f"{ratio:.2f}"
                text_bbox = cell_font.getbbox(cell_text, anchor="mm")  # left, top, left + width, top + height
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                if text_width <= cell_width and text_height <= cell_height:
                    text_color = (round(get_text_color((color[0] / 255, color[1] / 255, color[2] / 255))[0] * 255),) * 3
                    draw.text(s.scale(rect.center), cell_text, anchor="mm", font=cell_font, fill=text_color)

        # Draw centroids
        centroid = die.netlist.get_module(module.name).center
        assert centroid is not None
        draw.text(s.scale(centroid), "X", anchor="mm", font=medium_font)

        # Draw module subtitles
        if draw_text:
            assert dispersions is not None, "dispersions is required if draw_text is True"
            draw.text((s.x_offset, s.y_offset + grid_height + margin / 2),
                      f"{module.name}| A = {module.area():.2f} | D = {dispersions[module.name]:.2f}",
                      anchor="ls", font=medium_font, fill="Black")

        s.x_offset += grid_width + margin

    # Draw main title
    if draw_text:
        assert dispersions is not None, "dispersions is required if draw_text is True"
        assert alpha is not None, "alpha is required if draw_text is True"
        if len(suptitle) > 0:
            suptitle += " | "
        total_wire_length = die.netlist.wire_length
        total_dispersion = sum(dispersions.values())
        suptitle += f"alpha = {alpha:.2f} | " \
                    f"Wire length = {total_wire_length:.2f} | Total dispersion = {total_dispersion:.2f} | " \
                    f"Result = {(alpha * total_wire_length + (1 - alpha) * total_dispersion):.2f}"
        draw.text((img.width / 2, 0), suptitle, anchor="ma", font=large_font, fill="Black")

    return img


def do_plots(plotting_options: PlottingOptions, n_iter: int,
             die: Die, allocation: Allocation, dispersions: dict[str, float], alpha: float) -> None:
    """
    Create the plots according to the options given
    :param plotting_options: plotting options.
    :param n_iter: the iteration number (for the output filename).
    :param die: the die with the netlist.
    :param allocation: the allocation.
    :param dispersions: the dispersions of the modules.
    :param alpha: the value of the alpha hyperparameter.
    """
    assert die.netlist is not None, "No netlist associated to the die"

    if plotting_options.joint_plot:
        get_joint_floorplan_plot(die.netlist, die.bounding_box.shape, allocation). \
            save(f"{plotting_options.name}-joint-{n_iter}.gif")

    if plotting_options.separated_plot:
        get_separated_floorplan_plot(die, allocation, dispersions, alpha, draw_text=True). \
            save(f"{plotting_options.name}-separated-{n_iter}.png")
