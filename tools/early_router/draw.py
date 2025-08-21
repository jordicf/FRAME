from PIL import Image, ImageDraw
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point
from tools.draw.draw import (
    get_floorplan_plot,
    calculate_bbox,
    scale,
    calculate_scaling,
    Scaling,
    get_font,
)
from tools.early_router.hanan import HananGraph3D, HananNode3D
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from distinctipy import distinctipy
import matplotlib.patheffects as path_effects
from frame.netlist.netlist_types import HyperEdge, NamedHyperEdge
from tools.early_router.types import NetId, EdgeId, CellId
from matplotlib.patches import Circle, Rectangle as MplRectangle
import math
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.axes3d as Axes3D
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Any, Union
from frame.geometry.geometry import Shape, Rectangle
from tools.early_router.build_model import FeedThrough
import numpy as np

# Some colors
COLOR_GREY = (128, 128, 128)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


def scale_coordinates(x1, y1, x2, y2, scaling: Scaling, image_height: int):
    """
    Scales and converts coordinates from PIL to Matplotlib format.

    :param: x1, y1, x2, y2 Original rectangle coordinates in PIL (top-left origin)
    :param scaling: Scaling object with sx, sy factors
    :param image_height: Height of the original image (before scaling)
    :return: (x, y, width, height) formatted for Matplotlib
    """
    # Apply scaling factors
    scaled_x1 = x1 * scaling.xscale
    scaled_y1 = y1 * scaling.yscale
    scaled_x2 = x2 * scaling.xscale
    scaled_y2 = y2 * scaling.yscale

    # Convert to Matplotlib coordinates (bottom-left origin)
    scaled_y1 = scaling.yscale * (image_height - y2)  # Flip y
    scaled_y2 = scaling.xscale * (image_height - y1)  # Flip y

    # Compute width and height
    scaled_width = scaled_x2 - scaled_x1
    scaled_height = scaled_y2 - scaled_y1

    return scaled_x1, scaled_y1, scaled_width, scaled_height


def draw_solution3D(
    netlist: Netlist,
    route: list[dict[EdgeId, float]],
    net: NamedHyperEdge,
    hanan_graph: HananGraph3D,
    net_color: tuple[float, float, float, float],
    filepath: str = "plot3D",
    width=0,
    height=0,
    frame=50,
    fontsize=10,
):
    """
    Uses Matplotlib library to plot a route solution on the 3D space. The modules that must be connected by the route solution are seen in 3D with their name tags.

    :param netlist: Netlist class with all modules of the floorplan and list of hyperedges.
    :param route: List of dictionaries with keys edgeid corresponding to the segments to be routed.
    :param net: A HyperEdge class that contains the net information that is routed.
    :param hanan_graph: The hanan graph with nodes and edges, the edge positional points are retrieved from here.
    :param net_color: The color to draw the route of the net in float 0-1 values.
    :param filepath: string with the path and the name to save the frontal view of the routed plot.
    """
    if max(net_color) > 1:
        net_color = [
            (r / 255, g / 255, b / 255, o / 255) for (r, g, b, o) in [net_color]
        ][0]
    die_shape = calculate_bbox(netlist)
    scaling = calculate_scaling(die_shape, width, height, frame)
    # Create a 3D plot
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    # Assignment of a color to each block
    colors = distinctipy.get_colors(netlist.num_modules, rng=0)
    module2color = {b: colors[i] for i, b in enumerate(netlist.modules)}
    # Total image height
    total_im_h = scaling.height + 2 * frame
    # Setting limits
    x, y, width, height = scale_coordinates(
        0, 0, scaling.width + 2 * frame, scaling.height + 2 * frame, scaling, total_im_h
    )
    ax.set_xlim(width)
    ax.set_ylim(height)
    # Adding a frame
    x, y, width, height = scale_coordinates(
        frame / 2,
        frame / 2,
        scaling.width + 1.5 * frame,
        scaling.height + 1.5 * frame,
        scaling,
        total_im_h,
    )
    p: Any = MplRectangle(
        (x, y),
        width,
        height,
        edgecolor="black",
        fill=False,
        linewidth=3,
        alpha=0.3,
        zorder=1,
    )
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    modulenames = net.modules
    for m in netlist.modules:
        color = module2color[m]
        if m.num_rectangles == 0:
            radius = math.sqrt(m.area() / math.pi)
            assert m.center is not None
            c = scale(m.center, scaling)
            cx = c.x * scaling.xscale
            cy = (total_im_h - c.y) * scaling.yscale
            if m.is_iopin:
                p = Circle(
                    (cx, cy),
                    4 * total_im_h / len(module2color),
                    fc="black",
                    fill=True,
                    zorder=3,
                )
            else:
                p = Circle((cx, cy), radius, fc=color, fill=True, zorder=3)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
            if m.name in modulenames:
                ccolor = distinctipy.get_text_color(
                    (color[0], color[1], color[2]), threshold=0.6
                )
                ax.text(
                    cx,
                    cy,
                    0.1,
                    m.name,
                    fontsize=fontsize,
                    color="black",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground=ccolor)
                    ],
                    zorder=3,
                )  # Here error
        else:
            name = m.name
            for i, r in enumerate(m.rectangles):
                rname = name if m.num_rectangles == 1 else f"{name}[{i}]"
                bb = r.bounding_box
                ll = scale(bb.ll, scaling)
                ur = scale(bb.ur, scaling)
                x, y, width, height = scale_coordinates(
                    ll.x, ur.y, ur.x, ll.y, scaling, total_im_h
                )
                if name in modulenames:
                    ax.bar3d(
                        x,
                        y,
                        z=0,
                        dx=width,
                        dy=height,
                        dz=1,
                        color=color,
                        alpha=0.6,
                        zorder=1,
                    )
                    ccolor = distinctipy.get_text_color(
                        (color[0], color[1], color[2]), threshold=0.6
                    )
                    cx = x + width / 2
                    cy = y + height / 2
                    z = 0.5
                    ax.text(
                        cx,
                        cy,
                        z,
                        rname,
                        fontsize=fontsize,
                        color="black",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        path_effects=[
                            path_effects.withStroke(linewidth=1, foreground=ccolor)
                        ],
                        zorder=5,
                    )
                else:
                    p = MplRectangle(
                        (x, y),
                        width,
                        height,
                        fc=color,
                        fill=True,
                        visible=True,
                        alpha=0.3,
                        zorder=1,
                    )
                    ax.add_patch(p)
                    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    # Plot the route
    wl = 0.0
    cros = 0.0
    via = 0.0
    for i, edge_dict in enumerate(route):
        for (start, end), value in edge_dict.items():
            from_node = hanan_graph.get_node(start)
            to_node = hanan_graph.get_node(end)
            if not from_node or not to_node:
                continue
            canvas_start_p = scale(from_node.center, scaling)
            canvas_end_p = scale(to_node.center, scaling)
            x_vals, y_vals, z_vals = zip(
                (
                    canvas_start_p.x * scaling.xscale,
                    (total_im_h - canvas_start_p.y) * scaling.yscale,
                    from_node._id[2] + 0.01,
                ),
                (
                    canvas_end_p.x * scaling.xscale,
                    (total_im_h - canvas_end_p.y) * scaling.yscale,
                    to_node._id[2] + 0.01,
                ),
            )  # Extract coordinates
            ax.plot(
                x_vals,
                y_vals,
                z_vals,
                color=net_color,
                linewidth=3,
                zorder=4,
                path_effects=[path_effects.withStroke(linewidth=4, foreground="black")],
            )

            e = hanan_graph.get_edge(start, end)
            if not e:
                continue
            wl += e.length * value
            if (
                e.crossing
                and not (e.source.modulename in modulenames)
                and not (e.target.modulename in modulenames)
            ):
                cros += value
            if e.via:
                via += value

            s = f"{round(value, 1)}\n"
            ax.text(
                sum(x_vals) / 2,
                sum(y_vals) / 2,
                0.3 if min(z_vals) == 0.01 and max(z_vals) == 1.01 else sum(z_vals) / 2,
                s,
                fontsize=int(fontsize / 2) + 1,
                color="red",
                ha="center",
                va="center",
                path_effects=[path_effects.withStroke(linewidth=4, foreground="white")],
                zorder=5,
            )

    print(f"For net {net}. Total cost:\nWL={wl}\nMC={cros}\nVU={via}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Layers")
    plt.title("3D Route Visualization")
    ax.grid(False)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([0, 1], labels=["Horizontal", "Vertical"])  # Remove y-axis ticks
    plt.savefig(f"{filepath}.png")
    plt.close()
    return


def draw_solution2D(
    netlist: Netlist,
    routes: dict[NetId, list[dict[EdgeId, float]]],
    hanan_graph: HananGraph3D,
    color_pallete: dict[int, tuple[int, int, int, int]],
    width=0,
    height=0,
    frame=50,
    fontsize=20,
) -> Image.Image:
    die_shape = calculate_bbox(netlist)
    # Remove un-routed nets
    netlist._edges = []
    # Create base plot
    im = get_floorplan_plot(netlist, die_shape, None, width, height, frame, fontsize)

    scaling = calculate_scaling(die_shape, width, height, frame)
    # Create a Transparent layer for later merge
    transp = Image.new("RGBA", im.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(transp, "RGBA")

    drawn_edges: list = []
    line_width = 5
    for net_id, route in routes.items():
        for path in route:
            edge_id = list(path.keys())[0]
            other_way = edge_id[::-1]
            from_node = hanan_graph.get_node(edge_id[0])
            to_node = hanan_graph.get_node(edge_id[1])
            if not from_node or not to_node:
                continue

            canvas_start_p = scale(from_node.center, scaling)
            canvas_end_p = scale(to_node.center, scaling)

            if edge_id in drawn_edges or other_way in drawn_edges:
                # To know how much large the line has to be.
                n = drawn_edges.count(edge_id)
                m = drawn_edges.count(other_way)
                n = n + m
                drawing.line(
                    [
                        (
                            canvas_start_p.x + n * line_width,
                            canvas_start_p.y + n * line_width,
                        ),
                        (
                            canvas_end_p.x + n * line_width,
                            canvas_end_p.y + n * line_width,
                        ),
                    ],
                    fill=color_pallete[net_id],
                    width=line_width,
                )
                drawn_edges.append(edge_id)

            else:
                drawing.line(
                    [
                        (canvas_start_p.x, canvas_start_p.y),
                        (canvas_end_p.x, canvas_end_p.y),
                    ],
                    fill=color_pallete[net_id],
                    width=line_width,
                )
                drawn_edges.append(edge_id)

    im.paste(Image.alpha_composite(im, transp))
    return im


def create_canvas(s: Scaling):
    """
    Generates the canvas of the drawing.

    :param s: scaling of the layout
    """
    im = Image.new(
        "RGBA", (s.width + 2 * s.frame, s.height + 2 * s.frame), (255, 255, 255, 255)
    )
    drawing = ImageDraw.Draw(im)
    return im, drawing


def floorplan_plot(
    netlist: Netlist, die_shape: Shape, width: int = 0, height: int = 0, frame: int = 20
) -> Image.Image:
    """
    Draws only the outlines of modules on a blank canvas.
    """
    # Compute scaling for die
    scaling = calculate_scaling(die_shape, width, height, frame)
    # Create canvas
    im, drawing = create_canvas(scaling)

    # # Module outlines
    # for m in netlist.modules:
    #     for r in m.rectangles:
    #         bb = r.bounding_box
    #         # Scale rectangle corners and shift by frame
    #         ll = scale(bb.ll, scaling)
    #         ur = scale(bb.ur, scaling)
    #         drawing.rectangle((ll.x, ur.y, ur.x, ll.y), outline=COLOR_GREY, width=4)

    # Module outlines without drawing shared edges
    for m in netlist.modules:
        # Gather edges for each rectangle
        if m.rectangles:
            trunk = m.rectangles[0]
            for r in m.rectangles:
                bb = r.bounding_box
                # Scale and shift corners
                ll = scale(bb.ll, scaling)
                ur = scale(bb.ur, scaling)
                # Define corner points
                p_ll = (ll.x, ll.y)
                p_lr = (ur.x, ll.y)
                p_ur = (ur.x, ur.y)
                p_ul = (ll.x, ur.y)
                # Rectangle edges
                edges = [(p_ll, p_lr), (p_lr, p_ur), (p_ur, p_ul), (p_ul, p_ll)]
                loc = trunk.find_location(r)
                p = (-1, -1)
                if loc == Rectangle.StropLocation.NORTH:
                    a, b = edges.pop(0)
                    drawing.line(
                        [(a[0] + 2, a[1]), (b[0] - 2, b[1])], fill=COLOR_WHITE, width=4
                    )
                elif loc == Rectangle.StropLocation.SOUTH:
                    a, b = edges.pop(2)
                    drawing.line(
                        [(a[0] - 2, a[1]), (b[0] + 2, b[1])], fill=COLOR_WHITE, width=4
                    )
                elif loc == Rectangle.StropLocation.EAST:
                    a, b = edges.pop(1)
                    drawing.line(
                        [(a[0] + 2, a[1]), (b[0] - 2, b[1])], fill=COLOR_WHITE, width=4
                    )
                elif loc == Rectangle.StropLocation.WEST:
                    a, b = edges.pop(3)
                    drawing.line(
                        [(a[0] - 2, a[1]), (b[0] + 2, b[1])], fill=COLOR_WHITE, width=4
                    )
                for a, b in edges:
                    drawing.line([a, b], fill=COLOR_GREY, width=2)

    # Outer and inner frames
    drawing.rectangle(
        (0, 0, scaling.width + 2 * frame, scaling.height + 2 * frame),
        outline=COLOR_GREY,
        width=4,
    )
    drawing.rectangle(
        (frame, frame, scaling.width + frame, scaling.height + frame),
        outline=COLOR_BLACK,
        width=4,
    )
    return im


def congestion_to_color(value: float) -> Tuple[int, int, int, int]:
    """
    Map congestion percentage (0-100) to a green-to-red gradient.
    Returns an RGBA tuple with semi-transparency for overlap handling.
    """
    # Clamp and normalize
    norm = max(0.0, min(1.0, value / 100))
    red = int(norm * 255)
    green = int((1 - norm) * 255)
    opacity = 180  # Semi-transparent to help visualize overlaps
    return (red, green, 0, opacity)


def draw_congestion_legend(
    im: Image.Image,
    position: Tuple[int, int] = (20, 20),
    size: Tuple[int, int] = (200, 20),
    fontsize: int = 14,
) -> Image.Image:
    """
    Draws a horizontal color bar legend from green (0%) to red (100%)
    at the given position on the image.
    """
    width, height = size
    # Create gradient bar
    legend = Image.new("RGBA", (width, height + fontsize + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(legend)
    for i in range(width):
        pct = (i / (width - 1)) * 100
        color = congestion_to_color(pct)
        draw.line([(i, 0), (i, height)], fill=color)

    # Add text labels
    try:
        font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont] = ImageFont.truetype("arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    # "0%" at left
    draw.text((0, height + 2), "0%", fill="black", font=font)
    # "100%" at right
    text_w = draw.textlength("100%", font=font)
    draw.text((width - text_w, height + 2), "100%", fill="black", font=font)

    # Paste legend onto main image
    im.paste(legend, position, legend)
    return im


def draw_congestion(
    netlist: Netlist,
    edge_congestion: dict[EdgeId, float],
    hanan_graph: HananGraph3D,
    layer_id: list[int] = [],
    title: str = "Congestion Map",
    width: int = 0,
    height: int = 0,
    frame: int = 80,
    fontsize: int = 30,
) -> Image.Image:
    """
    Overlay congestion on module outlines and append a legend.
    Low-congestion lines drawn first; edges are semi-transparent.
    layer_id default drawing all
    """
    # Compute die bounding box and clear existing edges
    die_shape = hanan_graph.hanan_grid.shape
    netlist._edges = []

    # Base plot: module outlines only
    im = floorplan_plot(netlist, die_shape, width=width, height=height, frame=frame)

    # Pre-Process: No vias, no congestion with < 1 wire
    # Drawing all layers on a 2D
    edge_map: dict[tuple[CellId, CellId], tuple[float, float]] = {}
    for edge_id, congestion in edge_congestion.items():
        if congestion < 1:
            continue
        edge = hanan_graph.get_edge(edge_id[0], edge_id[1])
        from_node = hanan_graph.get_node(edge_id[0])
        to_node = hanan_graph.get_node(edge_id[1])
        if not from_node or not to_node or not edge:
            continue
        # Skip vias
        if from_node._id[-1] != to_node._id[-1]:
            continue
        elif layer_id and from_node._id[-1] in layer_id:
            # Skip non-layer selected
            continue

        sum_con, sum_cap = edge_map.get((from_node._id[:2], to_node._id[:2]), (0, 0))
        sum_con += congestion
        sum_cap += edge.capacity
        edge_map[(from_node._id[:2], to_node._id[:2])] = (sum_con, sum_cap)

    # Sort edges by raw congestion (low to high)
    # sorted_edges = sorted(edge_congestion.items(), key=lambda kv: kv[1])
    sorted_edges = sorted(edge_map.items(), key=lambda kv: kv[1][0] / kv[1][1])

    # Transparent overlay for drawing edges
    transp = Image.new("RGBA", im.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(transp, "RGBA")
    scaling = calculate_scaling(die_shape, width, height, frame)

    # Draw each edge
    for cells, tcong in sorted_edges:
        from_node = hanan_graph.get_node((cells[0][0], cells[0][1], 0))
        to_node = hanan_graph.get_node((cells[1][0], cells[1][1], 0))

        cell = hanan_graph.hanan_grid.get_cell(cells[0])
        if not from_node or not to_node or not cell:
            continue

        pct = (tcong[0] / tcong[1]) * 100
        if cell:
            if abs(from_node.center.x - to_node.center.x) < 1e-6:
                lw = pct * scaling.xscale * cell.width_capacity / 100
            elif abs(from_node.center.y - to_node.center.y) < 1e-6:
                lw = pct * scaling.yscale * cell.height_capacity / 100
        else:
            lw = pct * math.sqrt(
                scaling.xscale * scaling.yscale
            )  # Probably a terminal connection and will be 0
        lw = max(int(lw), 5)

        # Coordinates adjusted by frame
        start = scale(from_node.center, scaling)
        end = scale(to_node.center, scaling)
        start_xy = (start.x, start.y)
        end_xy = (end.x, end.y)

        # Draw line
        color = congestion_to_color(pct)
        drawing.line([start_xy, end_xy], fill=color, width=lw)

        # Annotate very high congestion (>99%)
        if pct > 99:
            s = f"{round(pct, 1)}%"
            font = get_font(int(fontsize / 2))
            l, t, r, b = drawing.multiline_textbbox((0, 0), s, font=font)
            txt_w, txt_h = r - l, b - t
            tx = round((start_xy[0] + end_xy[0]) / 2)
            ty = round((start_xy[1] + end_xy[1]) / 2)
            drawing.text(
                (tx, ty),
                s,
                fill="black",
                font=font,
                anchor="ms",
                stroke_width=1,
                stroke_fill="white",
            )

    if title:
        font = get_font(int(frame * 0.6))
        text_w = drawing.textlength(title, font=font)
        drawing.text(
            ((scaling.width + 2 * frame - text_w) / 2, int(frame * 0.2)),
            title,
            fill="black",
            font=font,
        )

    # Composite overlay and legend
    im = Image.alpha_composite(im.convert("RGBA"), transp)
    size = (int(scaling.width / 2), int(frame * 0.2))
    im = draw_congestion_legend(
        im,
        position=(
            int((scaling.width + 2 * frame - size[0]) / 2),
            int(scaling.height + frame * 1.2),
        ),
        size=size,
        fontsize=fontsize,
    )
    return im


def plot_net_distribution(ft: FeedThrough, filepath: str | None = None):
    """
    Produces a histogram of net weights vs. frequency.
    """
    data = [net.weight for net in ft._nets.values()]

    all_vals = np.array(data)
    all_vals = all_vals[all_vals > 0]  # Remove zeros or negatives if present

    bin_width = round((all_vals.max() - all_vals.min()) / 20, 2)
    min_log = np.floor(all_vals.min() / bin_width) * bin_width
    max_log = np.ceil(all_vals.max() / bin_width) * bin_width
    bin_edges = np.arange(min_log, max_log + bin_width, bin_width)

    # Compute histogram
    counts, edges = np.histogram(all_vals, bins=bin_edges)

    # Compute bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_widths = np.diff(edges)

    # Filter out bins with zero frequency
    mask = counts > 0
    counts = counts[mask]
    bin_centers = bin_centers[mask]
    bin_widths = bin_widths[mask]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=bin_widths, edgecolor="black", color="skyblue")

    # Label axes
    plt.xlabel("Net Weight")
    plt.ylabel("Frequency")
    plt.title("Histogram of Net Weight")

    # Customize x-ticks
    plt.xticks(bin_centers, [f"{b:.2f}" for b in bin_centers], rotation=45)
    plt.tight_layout()

    if filepath:
        plt.savefig(f"{filepath}.png")
    else:
        plt.show()
