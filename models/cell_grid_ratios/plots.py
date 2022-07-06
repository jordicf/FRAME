import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

rcParams['font.family'] = 'monospace'


def plot_result(block_areas: list[float], wire_costs: dict[tuple[int, int], float],
                ratios, centroids, dispersions, wire_length,
                n_rows: int, n_cols: int, cell_width: float, cell_height: float,
                suptitle: str | None = None, filename: str | None = None):
    """
    Plot a floorplan given the ratios of each block in each cell, and additional information to annotate the graphics.
    The plot is made up of subplots separated by each block. The block areas, centroids, dispersions and total wire
    length is also displayed. An optional title can also be provided, and a filename to save the plot to a file. If no
    filename is given, the plot is shown.
    """

    # TODO: Plot wires in some way.

    n_blocks = len(block_areas)

    # Create one subplot for each block, and an additional narrow one for the color bar
    fig, axs = plt.subplots(1, n_blocks + 1,
                            figsize=(n_blocks*n_cols/2, n_rows/2),
                            gridspec_kw=dict(width_ratios=([1.0]*n_blocks) + [0.1]))

    for area, block_configuration, centroid, dispersion, ax in zip(block_areas, ratios, centroids, dispersions, axs):
        matrix = np.around(block_configuration, 2)

        # Plot block configuration in cell grid with a heatmap
        x_pos, y_pos = np.meshgrid(np.linspace(0, n_cols*cell_width, n_cols + 1),
                                   np.linspace(0, n_rows*cell_height, n_rows + 1))
        mesh = ax.pcolormesh(x_pos, y_pos, matrix, vmin=0, vmax=1, cmap="Blues", zorder=0)

        # Annotate the heatmap
        mesh.update_scalarmappable()
        text_x_pos, text_y_pos = np.meshgrid(np.linspace(cell_width/2, n_cols*cell_width - cell_width/2, n_cols),
                                             np.linspace(cell_height/2, n_rows*cell_height - cell_height/2, n_rows))
        for x, y, color, val in zip(text_x_pos.flat, text_y_pos.flat, mesh.get_facecolors(), matrix.flat):
            text_kwargs = dict(color=".15" if sns.utils.relative_luminance(color) > 1/2 else "w", ha="center",
                               va="center")
            ax.text(x, y, val, **text_kwargs, zorder=2)

        # Plot the centroids with x makers
        ax.plot(centroid.x, centroid.y, marker='x', color="black", zorder=1)

        # Move the x-axis on the top and invert the y-axis to show the plot in matrix form
        ax.xaxis.tick_top()
        ax.invert_yaxis()

        ax.set_xlabel(f"Area = {area:.2f} | Dispersion = {dispersion:.2f}")

        ax.set_aspect("equal")

    # Plot the color bar
    fig.colorbar(axs[0].collections[0], cax=axs[n_blocks])

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
