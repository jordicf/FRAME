# GP-based Floorplan Optimization

A Geometric Programming (GP) approach for VLSI floorplanning with an iterative constraint relaxation strategy.

## Table of Contents
- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
- [Algorithm](#algorithm)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [References](#references)

---

## Overview

This tool solves the **floorplanning problem** using **Generalized Geometric Programming (GGP)**. The goal is to determine the positions and dimensions of rectangular modules on a chip while:
- Minimizing the total chip area (bounding box)
- Satisfying non-overlap constraints between modules
- Maintaining module area and aspect ratio requirements
- Respecting chip boundary constraints

### Key Features

- **Geometric Programming Formulation**: Converts the floorplanning problem into a GGP that can be solved efficiently
- **Constraint Relaxation Strategy**: Iteratively removes redundant constraints to improve solution quality
- **Slack-based Analysis**: Identifies and relaxes less critical constraints based on slack values
- **Flexible Module Shapes**: Allows modules to change aspect ratio within specified bounds

---

## Mathematical Formulation

### Decision Variables

For each module \(i = 1, \ldots, n\):
- \(x_i, y_i\): Center coordinates of module \(i\)
- \(w_i, h_i\): Width and height of module \(i\)
- \(W, H\): Bounding box width and height

### Objective Function

Minimize the chip area:

```
minimize W
subject to W / H = 1  (square aspect ratio)
```

### Constraints

#### 1. **Area Constraint**
Each module must maintain its specified area:

```
w_i × h_i = A_i,  ∀i
```

In GGP form (posynomial = 1):
```
(w_i × h_i) / A_i = 1
```

#### 2. **Aspect Ratio Constraint**
Limit the aspect ratio of each module to be within `[1/r_max, r_max]`:

```
w_i / (h_i × r_max) ≤ 1
h_i / (w_i × r_max) ≤ 1
```

where `r_max = 3.0` by default.

#### 3. **Boundary Constraints**
Modules must stay within the chip boundaries:

```
0.5 × w_i / x_i ≤ 1  (left boundary)
0.5 × h_i / y_i ≤ 1  (bottom boundary)
(x_i + 0.5×w_i) / W ≤ 1  (right boundary)
(y_i + 0.5×h_i) / H ≤ 1  (top boundary)
```

#### 4. **Non-overlap Constraints**

For each pair of modules \((i, j)\) that must not overlap:

**Horizontal Constraint Graph (HCG)**:
```
(x_i + 0.5×w_i + 0.5×w_j) / x_j ≤ 1
```
This enforces: module \(i\) is to the left of module \(j\).

**Vertical Constraint Graph (VCG)**:
```
(y_i + 0.5×h_i + 0.5×h_j) / y_j ≤ 1
```
This enforces: module \(i\) is below module \(j\).

### Why Geometric Programming?

All constraints are in the form of **posynomials** (sums of monomials with positive coefficients):
- Area constraints: `w_i × h_i / A_i = 1`
- Aspect ratio: `w_i / (h_i × r_max) ≤ 1`
- Non-overlap: `(x_i + 0.5×w_i + 0.5×w_j) / x_j ≤ 1`

This allows us to use efficient GP solvers that guarantee convergence to a **global optimum** (for convex GP) or a good local optimum (for Legalizer).

---

## Algorithm

### Constraint Graph Construction

The algorithm builds two constraint graphs based on the initial layout:

1. **Detect Overlaps**: For each module pair \((i, j)\), calculate:
   - Horizontal overlap: \(h\_overlap = \max(0, \min(x_i^{right}, x_j^{right}) - \max(x_i^{left}, x_j^{left}))\)
   - Vertical overlap: \(v\_overlap = \max(0, \min(y_i^{top}, y_j^{top}) - \max(y_i^{bottom}, y_j^{bottom}))\)

2. **Add Constraints**:
   - **No overlap** in both directions → Add to both HCG and VCG
   - **Only vertical overlap** → Add to HCG (separate horizontally)
   - **Only horizontal overlap** → Add to VCG (separate vertically)
   - **Both overlaps** → Add to the graph with larger overlap

### Iterative Constraint Relaxation

The algorithm performs multiple rounds of optimization:

```
for round = 1 to max_rounds:
    1. Setup GP model with current constraint graphs
    2. Solve GP optimization
    3. Calculate constraint slacks:
       - Horizontal slack: slack_h(i,j) = x_j - x_i
       - Vertical slack: slack_v(i,j) = y_j - y_i
    4. Identify redundant constraints:
       - Find module pairs with BOTH H and V constraints
       - For each such pair, keep the constraint with larger slack
       - Remove the constraint with smaller slack
    5. If no constraints removed, stop
```

**Rationale**: Module pairs with large slack have "extra space" between them. If both horizontal and vertical constraints exist for a pair, one is likely redundant. We remove the tighter one to give the optimizer more freedom in subsequent rounds.


---
## Dependencies

- **Python 3.7+**
- **GEKKO**: Optimization suite for solving GGP problems
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **PyYAML**: YAML file handling

---

## Usage

### Basic Command

```bash
python GP.py \
    --netlist <netlist.yaml> \
    --die <die.yaml> \
    --output <output.yaml> \
    --output-image <output.png>
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--netlist` | str | *required* | Input netlist YAML file |
| `--die` | str | *required* | Die information YAML file |
| `--output` | str | `output.yaml` | Output netlist YAML file |
| `--output-image` | str | `output.png` | Output floorplan image |
| `--output-dir` | str | `output` | Directory for intermediate results |
| `--max-iter` | int | `200` | Maximum GP solver iterations |
| `--max-ratio` | float | `3.0` | Maximum aspect ratio for modules |
| `--max-rounds` | int | `10` | Maximum constraint relaxation rounds |
| `--k-per-round` | int | `None` | Number of constraints to remove per round (None = remove all redundant) |

### Example

```bash
# Run optimization with default parameters
python GP.py \
    --netlist ami33_initial.yaml \
    --die ami33.die \
    --output ami33_optimized.yaml \
    --output-image ami33_optimized.png \
    --output-dir results

# Run with custom parameters
python GP.py \
    --netlist ami33_initial.yaml \
    --die ami33.die \
    --output ami33_optimized.yaml \
    --max-iter 300 \
    --max-ratio 2.5 \
    --max-rounds 15 \
    --k-per-round 10
```


### Generated Files

1. **Output netlist** (`output.yaml`): Final optimized layout
2. **Output image** (`output.png`): Visualization of final layout
3. **Intermediate results** (`output/round_*.yaml`, `output/round_*.png`): Results from each relaxation round

---

## Limitations

1. **Fixed Topology**: The constraint graphs are built from the initial layout and may not explore radically different topologies
2. **Local Optimum**: GGP may converge to a local optimum; multiple runs with different initial layouts may help
3. **Scalability**: Runtime grows with the number of modules and constraints (O(n²) constraints for n modules)

---

## References

1. **Geometric Programming for Design Optimization**
   - Boyd, S., Kim, S.-J., Vandenberghe, L., & Hassibi, A. (2007). "A tutorial on geometric programming."
   
2. **Floorplanning Algorithms**
   - Wang, T.-C., & Wong, D. F. (1992). "An optimal algorithm for floorplan area optimization."
   



