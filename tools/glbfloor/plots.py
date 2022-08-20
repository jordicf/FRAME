"""This file contains the code for plotting the floor plans"""

from dataclasses import astuple, dataclass
from math import ceil

from PIL import Image, ImageDraw, ImageFont
from distinctipy import get_text_color
from matplotlib import cm
import numpy as np

from frame.geometry.geometry import Point
from frame.netlist.netlist import Netlist
from frame.allocation.allocation import Allocation


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


def get_grid_image(netlist: Netlist, allocation: Allocation, dispersions: dict[str, tuple[float, float]] | None = None,
                   alpha: float | None = None, suptitle: str = "",
                   draw_borders: bool = True, draw_ratios: bool = True, draw_text: bool = True) -> Image.Image:
    """
    Return a PIL Image containing a floorplan plot, given the netlist and allocation of each module in each cell, and
    additional information for annotations.

    The plot is made up of separated subplots for each module. Module info (area, centroid, and dispersion) and model
    and floorplan info (alpha, total dispersion, total wire length, and objective function value) can also be displayed
    if provided. An optional title can be shown too.
    """
    margin = 40
    scale_factor = 150
    font_name = "arial.ttf"
    cell_font = ImageFont.truetype(font_name, round(scale_factor / 10))
    medium_font = ImageFont.truetype(font_name, round(scale_factor / 8))
    large_font = ImageFont.truetype(font_name, round(scale_factor / 6))
    color_map = cm.get_cmap("Blues")

    grid_width, grid_height = map(ceil, astuple(allocation.bounding_box.shape))
    grid_width *= scale_factor
    grid_height *= scale_factor

    n_modules = netlist.num_modules

    img = Image.new("RGB", (margin + (grid_width + margin) * n_modules, grid_height + 2 * margin), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    s = Scaling(scale_factor, margin, margin, grid_width, grid_height)

    for module in netlist.modules:
        for module_alloc in allocation.allocation_module(module.name):
            rect = allocation.allocation_rectangle(module_alloc.rect).rect
            unscaled_bbox_min, unscaled_bbox_max = rect.bounding_box
            bbox_min, bbox_max = s.scale(unscaled_bbox_min), s.scale(unscaled_bbox_max)

            ratio = module_alloc.area
            color = tuple((np.array(color_map(round(ratio * 255))) * 255).astype(np.uint8))

            # Draw cells
            draw.rectangle((bbox_min, bbox_max),
                           fill=color,  # type: ignore
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
        centroid = netlist.get_module(module.name).center
        assert centroid is not None
        draw.text(s.scale(centroid), "X", anchor="mm", font=medium_font)

        # Draw module subtitles
        if draw_text:
            assert dispersions is not None, "dispersions is required if draw_text is True"
            draw.text((s.x_offset, s.y_offset + grid_height + margin / 2),
                      f"{module.name}| A = {module.area():.2f} | D = {sum(dispersions[module.name]):.2f}",
                      anchor="ls", font=medium_font, fill="Black")

        s.x_offset += grid_width + margin

    # Draw main title
    if draw_text:
        assert dispersions is not None, "dispersions is required if draw_text is True"
        assert alpha is not None, "alpha is required if draw_text is True"
        if len(suptitle) > 0:
            suptitle += " | "
        total_wire_length = netlist.wire_length
        total_dispersion = sum(sum(d) for d in dispersions.values())
        suptitle += f"alpha = {alpha:.2f} | " \
                    f"Wire length = {total_wire_length:.2f} | Total dispersion = {total_dispersion:.2f} | " \
                    f"Result = {(alpha * total_wire_length + (1 - alpha) * total_dispersion):.2f}"
        draw.text((img.width / 2, 0), suptitle, anchor="ma", font=large_font, fill="Black")

    return img


def plot_grid(netlist: Netlist, allocation: Allocation, dispersions: dict[str, tuple[float, float]] | None = None,
              alpha: float | None = None, suptitle: str = "",
              draw_borders: bool = True, draw_ratios: bool = True, draw_text: bool = True,
              filename: str | None = None) -> None:
    """
    Plot the grid to a file given all the required parameters, or show it if no filename is given.
    See get_grid_image function documentation for more information.
    """
    img = get_grid_image(netlist, allocation, dispersions, alpha, suptitle, draw_borders, draw_ratios, draw_text)
    if filename is None:
        img.show()
    else:
        img.save(filename)
