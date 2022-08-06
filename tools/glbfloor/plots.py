import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from frame.geometry.geometry import Shape, Point
from frame.netlist.netlist import Netlist


def plot_grid(netlist: Netlist, ratios: list[list[list[float]]],
              centroids: list[Point], dispersions: list[float], wire_length: float,
              n_rows: int, n_cols: int, cell_shape: Shape,
              suptitle: str | None = None, filename: str | None = None):
    """
    Plot a floorplan given the ratios of each module in each cell, and additional information to annotate the graphics.
    The plot is made up of subplots separated by each module. The module areas, centroids, dispersions and total wire
    length is also displayed. An optional title can also be provided, and a filename to save the plot to a file. If no
    filename is given, the plot is shown.
    """

    # TODO: Plot wires in some way.

    n_modules = netlist.num_modules

    scaling_factor = 1 if n_cols > 4 else 2

    # Create one subplot for each module, and an additional narrow one for the color bar
    fig, axs = plt.subplots(1, n_modules + 1,
                            figsize=(n_modules * n_cols / scaling_factor, n_rows / scaling_factor),
                            gridspec_kw=dict(width_ratios=([1.0] * n_modules) + [0.1]))

    for module, module_configuration, centroid, dispersion, ax in zip(netlist.modules, ratios, centroids, dispersions,
                                                                      axs):
        matrix = np.around(module_configuration, 2)

        # Plot module configuration in cell grid with a heatmap
        x_pos, y_pos = np.meshgrid(np.linspace(0, n_cols * cell_shape.w, n_cols + 1),
                                   np.linspace(0, n_rows * cell_shape.h, n_rows + 1))
        mesh = ax.pcolormesh(x_pos, y_pos, matrix, vmin=0, vmax=1, cmap="Blues", zorder=0)

        # Annotate the heatmap
        mesh.update_scalarmappable()
        text_x_pos, text_y_pos = np.meshgrid(
            np.linspace(cell_shape.w / 2, n_cols * cell_shape.w - cell_shape.w / 2, n_cols),
            np.linspace(cell_shape.h / 2, n_rows * cell_shape.h - cell_shape.h / 2, n_rows))
        for x, y, color, val in zip(text_x_pos.flat, text_y_pos.flat, mesh.get_facecolors(), matrix.flat):
            text_kwargs = dict(color=".15" if sns.utils.relative_luminance(color) > 1 / 2 else "w", ha="center",
                               va="center")
            ax.text(x, y, val, **text_kwargs, zorder=2)

        # Plot the centroids with x makers
        ax.plot(centroid.x, centroid.y, marker='x', color="black", zorder=1)

        # Move the x-axis to the top
        ax.xaxis.tick_top()

        ax.set_xlabel(f"{module.name}| A = {module.area():.2f} | D = {dispersion:.2f}")

        ax.set_aspect("equal")

    # Plot the color bar
    fig.colorbar(axs[0].collections[0], cax=axs[n_modules])

    # General plot title
    if suptitle is None:
        suptitle = ""
    else:
        suptitle += " | "
    suptitle += f"Wire length = {wire_length:.2f} | Total dispersion = {sum(dispersions):.2f}"
    plt.suptitle(suptitle)

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
