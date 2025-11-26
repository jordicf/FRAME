## Attraction-Repulsion based floorplan

Relocates and reshapes non-fixed modules using the attraction-repulsion algorithm.

### Overview

The purpose of this tool is to generate initial layouts for other tools, to improve the quality of their results. The solutions that are found are not legal, but
overlap is significantly smaller than with graph drawing methods.

It takes as input a die file, a netlist file, which must have initialized positions, and it produces a new netlist file, with positions and shapes changed. Note that this tool produces *deterministic* results.

### Usage

#### Basic Command

```bash
frame attrep \
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
| `--stopping_tolerance` | float | 1e-4 | Minimum relative change to stop the main loop (the higher the earlier stop) |
| `--overlap_tolerance` | float | 1e-3 | Minimum relative change to stop the main expansion phase (the lower, the less overlap but more cpu time) |

### Algorithm

The algorithm iterates

### Dependencies
- **Python 3**
- **Numpy**
- **NetworkX**