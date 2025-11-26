## Graph drawing based floorplan

Relocates non-fixed modules using a modified version of the  ForceAtlas2 graph drawing algorithm. 

### Overview

The purpose of this tool is to generate initial layouts for other tools, to improve the quality of their results. It does not guarantee legal solutions, as it generates a lot of overlap.

It takes as input a die file, a netlist file, which can have initialized positions or not, and it produces a new netlist file, changing only the locations of the modules, but not their shapes. In the case of modules having no defined shape in the input, they are treated as squares.

This tool produces *deterministic* results, unless at least one module position is uninitialized and no seed parameter is set.

### Usage

#### Basic Command

```bash
frame grdraw \
    --netlist <netlist.yaml> \
    --die <die.yaml> \
    --output <output.yaml>
```

#### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--netlist` | str | *required* | Input netlist YAML file |
| `--die` | str | *required* | Die information YAML file |
| `--output` | str | *required* | Output netlist YAML file |
| `--max_iter` | int | 100 | Maximum iterations for the FA2 algorithm |
| `--seed` | int | None | Random seed for initial positions; has no effect if the netlist file already has initialized positions |
| `--scaling_factor` | float | None | Ratio of the die size vs FA2 layout size. If provided, scaling estimation phase is skipped, saving ~50% time |
| `--scaling_factor_read` | str | None | Same as `--scaling_factor` but in a file. If both `--scaling_factor` and `--scaling_factor_read` are provided, `--scaling_factor_read` gets ignored. |
| `--scaling_factor_write` | str | None | File to write the scaling factor, to save time in fututure runs.|

### Algorithm

Modules are represented as vertices in a graph, and nets as edges. Multi-module nets (hyperedges) are handled by introducing dummy nodes, connecting each module in the net to the dummy node.

The layout algorithm is force-directed. Each iteration attractive forces are calculated for each edge, as well as repulsive forces between every pair of nodes. Larger modules exert a higher repulsive force, as they need more empty space around them.

#### Fixed Modules

Fixed modules contribute to the force computations, but they are not moved. This presents a problem, as the coordinates of the fixed vertices may not be of the same magnitude as the rest. In general, it is very hard to tell how large the bounding box of a ForceAtlas2 layout will be, as it varies a lot depending on the number of vertices, sparsity of the graph and its overall structure.

To address this, a two-phase approach is used:

**Phase 1**: Ignore fixed nodes and run ForceAtlas2 to estimate the natural scale of the layout.

**Phase 2**: Scale all coordinates (including fixed ones) so that the die size matches the estimated scale. Then rerun ForceAtlas2, this time keeping fixed nodes stationary, and undo the scaling.

The scaling factor needs only to be computed once, and can be reused in later runs. It can be set with the `--scaling_factor` argument.

#### Initialization

ForceAtlas2 requires an initial position for each node:

- If positions are not defined in the input netlist, a random layout is assigned (optionally using `--seed`).

- If positions are already defined, they are used as the starting layout, and `--seed` is ignored.

### Dependencies
- **Python 3**
- **Numpy**
- **NetworkX**