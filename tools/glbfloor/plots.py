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
    scale_factor: float
    x_offset: int
    y_offset: int
    grid_width: int
    grid_height: int

    def scale(self, p: Point) -> tuple[int, int]:
        return (round(p.x * self.scale_factor + self.x_offset),
                round(self.grid_height - p.y * self.scale_factor + self.y_offset))


def plot_grid(netlist: Netlist, allocation: Allocation, dispersions: dict[str, float],
              suptitle: str | None = None, filename: str | None = None) -> None:
    """
    Plot a floorplan given the netlist and allocation of each module in each cell, and additional information to
    annotate the graphics.

    The plot is made up of subplots separated by each module. The module areas, centroids, dispersions and total wire
    length are also displayed. An optional title can also be provided, and a filename to save the plot to a file. If no
    filename is given, the plot is shown.
    """
    margin = 40
    scale_factor = 150
    cell_font = ImageFont.truetype("arial.ttf", round(scale_factor / 10))
    subtitle_font = ImageFont.truetype("arial.ttf", round(scale_factor / 8))
    suptitle_font = ImageFont.truetype("arial.ttf", round(scale_factor / 6))
    color_map = cm.get_cmap("Blues")

    grid_width, grid_height = map(ceil, astuple(allocation.bounding_box.shape))
    grid_width *= scale_factor
    grid_height *= scale_factor

    n_modules = netlist.num_modules

    img = Image.new("RGB", (margin + (grid_width + margin) * n_modules, grid_height + 2 * margin), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    s = Scaling(scale_factor, margin, margin, grid_width, grid_height)

    for i, module in enumerate(netlist.modules):
        for module_alloc in allocation.allocation_module(module.name):
            rect = allocation.allocation_rectangle(module_alloc.rect).rect
            bbox_min, bbox_max = rect.bounding_box
            ratio = module_alloc.area

            color = tuple((np.array(color_map(round(ratio * 255))) * 255).astype(np.uint8))
            draw.rectangle((s.scale(bbox_min), s.scale(bbox_max)), fill=color, outline="Black")  # type: ignore

            text_color = (round(get_text_color((color[0] / 255, color[1] / 255, color[2] / 255))[0] * 255),) * 3
            draw.text(s.scale(rect.center), f"{ratio:.2f}", anchor="mm", font=cell_font, fill=text_color)

        centroid = netlist.get_module(module.name).center
        assert centroid is not None
        draw.text(s.scale(centroid), "X", anchor="mm", font=cell_font)

        draw.text((s.x_offset, s.y_offset + grid_height + margin / 2),
                  f"{module.name}| A = {module.area():.2f} | D = {dispersions[module.name]:.2f}",
                  anchor="ls", font=subtitle_font, fill="Black", )

        s.x_offset += grid_width + margin

    if suptitle is None:
        suptitle = ""
    else:
        suptitle += " | "
    suptitle += f"Wire length = {netlist.wire_length:.2f} | Total dispersion = {sum(dispersions.values()):.2f}"
    draw.text((img.width / 2, 0), suptitle, anchor="ma", font=suptitle_font, fill="Black")

    if filename is None:
        img.show()
    else:
        img.save(filename)
