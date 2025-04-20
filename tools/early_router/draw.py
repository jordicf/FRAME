from PIL import Image, ImageDraw
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point
from tools.draw.draw import get_floorplan_plot, calculate_bbox, scale, calculate_scaling, Scaling, get_font
from tools.early_router.build_model import HananGraph3D
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from distinctipy import distinctipy
import matplotlib.patheffects as path_effects
from frame.netlist.netlist_types import HyperEdge
from tools.early_router.types import NetId, EdgeID
from matplotlib.patches import Circle, Rectangle
import math
import mpl_toolkits.mplot3d.art3d as art3d


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


def draw_solution3D(netlist: Netlist, route: list[dict[EdgeID, float]],net:HyperEdge, 
                    hanan_graph:HananGraph3D, net_color:tuple[float, float, float, float], filepath:str="plot3D", 
                    width=0, height=0, frame=50, fontsize=10):
    """
    Uses Matplotlib library to plot a route solution on the 3D space. The modules that must be conected by the route solution are seen in 3D with their name tags.

    :param netlist: Netlist class with all modules of the floorplan and list of hyperedges.
    :param route: List of dictiornaries with keys edgeid corresponding to the segments to be routed.
    :param net: A HyperEdge class that contains the net information that is routed.
    :param hanan_graph: The hanan graph with nodes and edges, the edge positional points are retrieved from here.
    :param net_color: The color to draw the route of the net in float 0-1 values.
    :param filepath: string with the path and the name to save the frontal view of the routed plot.
    """
    if max(net_color) > 1:
        net_color = [(r/255,g/255,b/255,o/255) for (r,g,b,o) in [net_color]][0]
    die_shape = calculate_bbox(netlist)
    scaling = calculate_scaling(die_shape, width, height, frame)
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Assignment of a color to each block
    colors = distinctipy.get_colors(netlist.num_modules, rng=0)
    module2color = {b: colors[i] for i, b in enumerate(netlist.modules)}
    # Total image height
    total_im_h = scaling.height + 2 * frame
    # Setting limits
    x, y, width, height = scale_coordinates(0, 0, scaling.width + 2 * frame, scaling.height + 2 * frame, scaling, total_im_h)
    ax.set_xlim(width)
    ax.set_ylim(height)
    # Adding a frame
    x, y, width, height = scale_coordinates(frame/2,frame/2, scaling.width + 1.5*frame, scaling.height + 1.5*frame, scaling, total_im_h)
    p=Rectangle((x,y), width, height, edgecolor='black', fill=False, linewidth=3, alpha=0.3, zorder=1)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    modulenames = [m.name for m in net.modules]
    for m in netlist.modules:
            color = module2color[m]
            if m.num_rectangles == 0:
                radius = math.sqrt(m.area() / math.pi)
                assert m.center is not None
                c = scale(m.center, scaling)
                cx = c.x * scaling.xscale
                cy = (total_im_h - c.y)* scaling.yscale
                if m.is_terminal:
                    p=Circle((cx,cy),4*total_im_h/len(module2color), fc='black', fill=True, zorder=3)
                else:
                    p=Circle((cx,cy),radius, fc=color, fill=True, zorder=3)
                ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
                if m.name in modulenames:
                    ccolor = distinctipy.get_text_color((color[0], color[1], color[2]), threshold=0.6)
                    ax.text(cx, cy, 0.1, m.name, fontsize=fontsize, color='black', ha="center", va="center", fontweight='bold',
                                   path_effects=[path_effects.withStroke(linewidth=1, foreground=ccolor)], zorder=3) # Here error
            else:
                name = m.name
                for i, r in enumerate(m.rectangles):
                    rname = name if m.num_rectangles == 1 else f"{name}[{i}]"
                    bb = r.bounding_box
                    ll = scale(bb.ll, scaling)
                    ur = scale(bb.ur, scaling)
                    x, y, width, height = scale_coordinates(ll.x, ur.y, ur.x, ll.y,scaling, total_im_h)
                    if name in modulenames:
                        ax.bar3d(x, y, z=0, dx=width, dy=height, dz=1, color=color, alpha=0.6, zorder=1)
                        ccolor = distinctipy.get_text_color((color[0], color[1], color[2]), threshold=0.6)
                        cx=x+width/2
                        cy=y+height/2
                        z=0.5
                        ax.text(cx, cy, z, rname, fontsize=fontsize, color='black', ha="center", va="center", fontweight='bold',
                                   path_effects=[path_effects.withStroke(linewidth=1, foreground=ccolor)], zorder=5)
                    else:
                        p = Rectangle((x,y), width, height, fc=color, fill=True, visible=True, alpha=0.3, zorder=1)
                        ax.add_patch(p)
                        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    # Plot the route
    wl = 0
    cros =0
    via = 0
    for i, edge_dict in enumerate(route):
        for (start, end), value in edge_dict.items():
            from_node = hanan_graph.get_node(start)
            to_node = hanan_graph.get_node(end)
            canvas_start_p = scale(from_node.center, scaling)
            canvas_end_p = scale(to_node.center, scaling)
            x_vals, y_vals, z_vals = zip(
                (canvas_start_p.x * scaling.xscale,
                 (total_im_h - canvas_start_p.y)* scaling.yscale,
                 from_node._id[2]+0.01), 
                (canvas_end_p.x * scaling.xscale,
                 (total_im_h - canvas_end_p.y)* scaling.yscale,
                 to_node._id[2]+0.01))  # Extract coordinates
            ax.plot(x_vals, y_vals, z_vals, color=net_color, linewidth=3, zorder=4,
                    path_effects=[path_effects.withStroke(linewidth=4, foreground='black')])
            
            e = hanan_graph.get_edge(start,end)
            wl += e.length * value
            avoidable_crossing = False
            if e.crossing and not(e.source.modulename in modulenames) and not(e.target.modulename in modulenames):
                avoidable_crossing = True
            if avoidable_crossing:
                cros += value
            if e.via:
                via += value

            s = f"{round(value,1)}\n"
            ax.text(sum(x_vals)/2, sum(y_vals)/2, 0.3 if min(z_vals)==0.01 and max(z_vals) == 1.01 else sum(z_vals)/2, s, 
                    fontsize=int(fontsize/2)+1, color='red', ha="center", va="center", 
                    path_effects=[path_effects.withStroke(linewidth=4, foreground='white')],zorder=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Layers")
    plt.title("3D Route Visualization")
    ax.grid(False)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([0,1],labels=['Horizontal','Vertical'])  # Remove y-axis ticks
    plt.savefig(f"{filepath}.png")
    plt.close()
    return


def draw_solution2D(netlist: Netlist, routes: dict[NetId, list[dict[EdgeID, float]]], 
                  hanan_graph: HananGraph3D, color_pallete: dict[int, tuple[float,float,float,float]],
                  width=0, height=0, frame=50, fontsize=20)->Image.Image:
    
    die_shape = calculate_bbox(netlist)
    # Remove un-routed nets
    netlist._edges = []
    # Create base plot
    im = get_floorplan_plot(netlist, die_shape, None, width, height, frame, fontsize)
    
    scaling = calculate_scaling(die_shape, width, height, frame)
    # Create a Transparent layer for later merge
    transp = Image.new('RGBA', im.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(transp, "RGBA")
    
    drawn_edges = []
    line_width = 5
    for net_id, route in routes.items():
        for path in route:
            edge_id = list(path.keys())[0]
            other_way = edge_id[::-1]
            from_node = hanan_graph.get_node(edge_id[0])
            to_node = hanan_graph.get_node(edge_id[1])

            canvas_start_p = scale(from_node.center, scaling)
            canvas_end_p = scale(to_node.center, scaling)

            if edge_id in drawn_edges or other_way in drawn_edges:
                # To know how much large the line has to be.
                n = drawn_edges.count(edge_id)
                m = drawn_edges.count(other_way)
                n = n+m
                drawing.line([(canvas_start_p.x + n*line_width, canvas_start_p.y + n*line_width),
                            (canvas_end_p.x + n*line_width, canvas_end_p.y + n*line_width)], 
                            fill=color_pallete[net_id], width= line_width)
                drawn_edges.append(edge_id)
                
            else:
                drawing.line([(canvas_start_p.x,canvas_start_p.y),(canvas_end_p.x,canvas_end_p.y)], 
                    fill=color_pallete[net_id], width= line_width)
                drawn_edges.append(edge_id)

    im.paste(Image.alpha_composite(im, transp))
    return im


def draw_congestion(netlist: Netlist, edge_congestion: dict[EdgeID,float], hanan_graph: HananGraph3D,
                  width=0, height=0, frame=50, fontsize=20)->Image.Image:
    # Do a cumsum for each edge. Comapre it to the edge capacity to get the %
    die_shape = calculate_bbox(netlist)
    # Remove un-routed nets
    netlist._edges = []
    # Create base plot
    im = get_floorplan_plot(netlist, die_shape, None, width, height, frame, fontsize)
    
    scaling = calculate_scaling(die_shape, width, height, frame)
    # Create a Transparent layer for later merge
    transp = Image.new('RGBA', im.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(transp, "RGBA")

    for edge_id, congestion in edge_congestion.items():
        edge = hanan_graph.get_edge(edge_id[0], edge_id[1])
        from_node = hanan_graph.get_node(edge_id[0])
        to_node = hanan_graph.get_node(edge_id[1])

        congestion_percentage = (congestion/edge.capacity) *100

        if from_node.center == to_node.center:
            # it is a via
            continue
        if abs(from_node.center.x - to_node.center.x) < 1e-6:
            line_width = congestion * scaling.xscale
        elif abs(from_node.center.y - to_node.center.y) < 1e-6:
            line_width = congestion * scaling.yscale
        else:
            # Terminal to die
            line_width = congestion * math.sqrt(scaling.xscale*scaling.yscale)
        if line_width < 5:
            line_width=5
        canvas_start = scale(from_node.center, scaling)
        canvas_end = scale(to_node.center, scaling)
        color = congestion_to_color(congestion_percentage)
        drawing.line([(canvas_start.x,canvas_start.y),(canvas_end.x,canvas_end.y)], 
                fill=color, width= int(line_width))
        
        if congestion_percentage > 99:
            s =f"{round(congestion_percentage,1)}%"
            # # Draw the congestion percentage
            font = get_font(fontsize)
            left, top, right, bottom = drawing.multiline_textbbox(xy=(0, 0), text=s, font=font)  # To center the text
            txt_w, txt_h = right - left, bottom - top
            txt_x, txt_y = round((canvas_start.x + canvas_end.x) / 2), round((canvas_start.y + canvas_end.y) / 2)
            drawing.text((txt_x, txt_y), s, fill='black', font=font, align="center",
                        anchor="ms", stroke_width=1, stroke_fill='white')

        
    im.paste(Image.alpha_composite(im, transp))
    return im


def congestion_to_color(value:float) -> Tuple[int,int,int,int]:
    # Normalize value to range [0, 1]
    normalized = value / 100
    # Map to red gradient (light to dark)
    red = int(183 + (normalized * (255 - 183)))  # Darker reds for higher congestion
    green = int(235 - (normalized * 235))       # Reduce green as congestion increases
    blue = int(238 - (normalized * 238))        # Reduce blue as congestion increases
    #opacity = 128
    opacity = 255
    return (red, green, blue, opacity)
