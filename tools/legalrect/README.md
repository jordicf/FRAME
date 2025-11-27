# FRAME Legalizer Tools

## Overview

The `tools/legalizer` directory provides a comprehensive suite of module legalization tools for VLSI/IC physical design. These tools transform an initial placement into a legal, overlap-free, and constraint-satisfying layout. The toolkit includes both **non-convex optimization solvers** and **convex optimization solvers** (via Geometric Programming), each with distinct optimization objectives and characteristics.

---

## Tool Classification

### Non-Convex Optimization Solvers

These solvers use general-purpose nonlinear programming (NLP) to handle complex, non-convex floorplanning objectives and constraints:

1. **`legalizer.py`** - GEKKO-based local/global legalizer
2. **`glb_legalizer.py`** - CasADi-based global legalizer

### Convex Optimization Solvers (Geometric Programming)

These solvers transform the legalization problem into a convex optimization problem by using Geometric Programming (GP) in log-space:

1. **`gp_area_legalizer.py`** - Minimize total bounding box area
2. **`gp_wl_legalizer.py`** - Minimize relative stretch (wirelength proxy)

---

## 1. Non-Convex Optimization Solvers

### 1.1 `legalizer.py` - GEKKO-based Legalizer

**Interface:** GEKKO (Python wrapper for APMonitor)  
**Solver:** APOPT(Linux, MACOS) , IPOPT(Windows)
**Flexibility:** Local/Global legalization modes

#### Key Features

- **Soft/Hard/Fixed Module Support**: Handles resizable soft modules, fixed-size hard modules, and fixed-position terminals
- **Rectilinear Modules**: Full support for trunk+branches structures (L-shape, T-shape, etc.)
- **Expression Tree Framework**: Symbolic constraint and objective modeling via `expr_tree.py`
- **Model Wrapper**: High-level optimization interface via `modelwrap.py`
- **Two Modes**:
  - **Global Legalization** (default): Large movements, global optimization
  - **Local Legalization** (`--small_steps`): Small incremental movements for fine-tuning

#### Command-Line Arguments

```bash
python tools/legalizer/legalizer.py <netlist.yaml> <die.yaml> [options]
```

**Basic Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `netlist.yaml` | positional | required | Input netlist file |
| `die.yaml` | positional | required | Input die specification |
| `--outfile FILE` | string | None | Output result YAML file |
| `--max_ratio FLOAT` | float | 3.0 | Maximum aspect ratio for modules |
| `--num_iter INT` | int | 15 | Number of optimization iterations |
| `--wl_mult FLOAT` | float | 1.0 | Wirelength objective weight multiplier |
| `--verbose` | flag | False | Enable detailed debug output |
| `--plot` | flag | False | Generate layout visualization per iteration |

**Advanced Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--small_steps` | flag | False | **Enable local legalization mode** (fine-grained movement constraints) |
| `--radius FLOAT` | float | 1.0 | Movement radius constraint (active set radius) |
| `--tau_initial FLOAT` | float | auto | Initial tau for soft constraints (larger = softer) |
| `--tau_decay FLOAT` | float | 0.9 | Tau decay factor per iteration (progressive tightening) |
| `--otol_initial FLOAT` | float | 1e-1 | Initial objective tolerance |
| `--otol_final FLOAT` | float | 1e-4 | Final objective tolerance |
| `--rtol_initial FLOAT` | float | 1e-1 | Initial constraint violation tolerance |
| `--rtol_final FLOAT` | float | 1e-4 | Final constraint violation tolerance |
| `--tol_decay FLOAT` | float | 0.5 | Tolerance decay factor per iteration |

**Example Usage:**

```bash
# Global legalization with visualization
python tools/legalizer/legalizer.py \
  bench-exam/MCNC/ami33.netlist.yaml \
  bench-exam/MCNC/ami33.die.yaml \
  --plot --num_iter 20 --outfile result.yaml

# Local legalization (fine-tuning mode)
python tools/legalizer/legalizer.py \
  initial_placement.yaml die.yaml \
  --small_steps --radius 0.3 --num_iter 30 --outfile refined.yaml
```

#### Supporting Files

- **`expr_tree.py`**: Expression tree for symbolic constraint modeling
  - `ExpressionTree` class: Symbolic expressions with automatic differentiation
  - `Equation` class: Constraint representation (equality/inequality)
  - `Cmp` enum: Comparison operators (EQ, LE, GE)
  
- **`modelwrap.py`**: GEKKO wrapper for optimization
  - `ModelWrapper` class: High-level interface to GEKKO
  - Methods: `build_model()`, `solve()`, `verify()`, `add_constraint()`
  - Features: Dynamic constraint adjustment, variable fixing

---

### 1.2 `glb_legalizer.py` - CasADi-based Global Legalizer

**Interface:** CasADi (Symbolic Framework for Numerical Optimization)  
**Solver:** Ipopt (Linux, MACOS, Windows) 
**Specialty:** Fast global legalization with active set optimization

#### Key Features

- **Smooth NLP Formulation**: All constraints are smooth and differentiable
- **Soft Constraints with τ**: Uses `smax` smoothing for no-overlap constraints
- **Active Set Optimization**: Selectively activates constraints based on module proximity
- **Rectilinear Module Support**: Full trunk+branches support with attachment constraints
- **Efficient Scaling**: Handles 100+ module designs efficiently
- **Visualization**: Per-iteration layout plots with HPWL/Overlap metrics

#### Command-Line Arguments

```bash
python tools/legalizer/glb_legalizer.py <netlist.yaml> <die.yaml> [options]
```

**Core Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `netlist.yaml` | positional | required | Input netlist file |
| `die.yaml` | positional | required | Input die specification |
| `--outfile FILE` | string | None | Output result YAML file |
| `--max_ratio FLOAT` | float | 3.0 | Maximum aspect ratio for modules |
| `--num_iter INT` | int | 15 | Number of optimization iterations |
| `--verbose` | flag | False | Enable detailed solver output |

**Performance Tuning:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--radius FLOAT` | float | 1.0 | **Active set radius** (0.0-1.0): <br>Controls which module pairs get non-overlap constraints<br>• 1.0 = all pairs (992 constraints)<br>• 0.5 = nearby pairs only (709 constraints, -28%)<br>• 0.2 = close pairs only (344 constraints, -65%) |
| `--tau_initial FLOAT` | float | auto | Initial τ for soft constraints (smoothing parameter) |
| `--tau_decay FLOAT` | float | 0.3 | τ decay factor (τ decreases over iterations) |

**Solver Tolerances:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--otol_initial FLOAT` | float | 1e-1 | Initial objective tolerance for Ipopt |
| `--otol_final FLOAT` | float | 1e-4 | Final objective tolerance |
| `--rtol_initial FLOAT` | float | 1e-1 | Initial constraint violation tolerance |
| `--rtol_final FLOAT` | float | 1e-4 | Final constraint violation tolerance |
| `--tol_decay FLOAT` | float | 0.5 | Tolerance decay factor per iteration |

**Visualization:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--plot` | flag | False | **Enable per-iteration visualization**<br>Generates PNG images showing layout evolution |
| `--plot_dir DIR` | string | "plots" | Directory for saving visualization images |

**Other:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wl_mult FLOAT` | float | 1.0 | Wirelength objective weight multiplier |
| `--palette_seed INT` | int | None | Random seed for color palette |

**Example Usage:**

```bash
# Fast global legalization with active set (65% fewer constraints)
python tools/legalizer/glb_legalizer.py \
  netlist.yaml die.yaml \
  --outfile result.yaml \
  --radius 0.2 --num_iter 20

# Full constraints with visualization
python tools/legalizer/glb_legalizer.py \
  netlist.yaml die.yaml \
  --outfile result.yaml \
  --radius 1.0 --num_iter 10 \
  --plot --plot_dir optimization_plots --verbose

# Multi-stage: coarse → fine
# Stage 1: Large movements, loose constraints
python tools/legalizer/glb_legalizer.py \
  netlist.yaml die.yaml --outfile stage1.yaml \
  --radius 0.3 --tau_initial 0.01 --num_iter 15

# Stage 2: Refinement with all constraints
python tools/legalizer/glb_legalizer.py \
  stage1.yaml die.yaml --outfile final.yaml \
  --radius 1.0 --tau_initial 0.0001 --num_iter 20
```

#### Technical Details

**Soft Constraint Formulation:**
```python
# No-overlap constraint: smax(t1, t2, τ) ≥ 0
# where:
#   t1 = (xi - xj)² - 0.25(wi + wj)²  (horizontal separation)
#   t2 = (yi - yj)² - 0.25(hi + hj)²  (vertical separation)
#   smax(a, b, τ) = 0.5(a + b + √((a-b)² + 4τ²))
#
# As τ → 0: smax → max(a, b) (hard constraint)
```

**Active Set Strategy:**
- Only applies non-overlap constraints to module pairs within distance threshold
- Distance threshold = `radius × max(die_width, die_height)`
- Dramatically reduces constraint count for large designs
- Example: For 28 modules:
  - radius=1.0: 398 active pairs, 992 rectangle-pair constraints
  - radius=0.5: 289 active pairs, 709 constraints (-28%)
  - radius=0.2: 143 active pairs, 344 constraints (-65%)

**Rectilinear Module Support:**
- Each module can have multiple rectangles (1 trunk + N branches)
- Attachment constraints ensure branches stay connected to trunk
- Intra-module non-overlap prevents branches from overlapping
- Inter-module non-overlap checks ALL rectangle pairs between modules

---

## 2. Convex Optimization Solvers (Geometric Programming)

### Mathematical Background

**Geometric Programming (GP)** is a special class of optimization problems that can be transformed into **convex problems** by applying a logarithmic change of variables. The key advantage is **global optimality** guarantee and fast convergence.

**GP Standard Form:**
```
minimize    f₀(x)
subject to  fᵢ(x) ≤ 1,  i = 1, ..., m
            gⱼ(x) = 1,   j = 1, ..., p
```

where `f` are posynomials (sums of monomials with positive coefficients) and `g` are monomials.

**Log-Space Transformation:**
By substituting `xᵢ = exp(yᵢ)`, the problem becomes:
```
minimize    log(f₀(exp(y)))
subject to  log(fᵢ(exp(y))) ≤ 0
            log(gⱼ(exp(y))) = 0
```

This is a **convex optimization problem** in log-space, solvable by interior-point methods with polynomial-time complexity.

---

### 2.1 `gp_area_legalizer.py` - Area Minimization via GP

**Objective:** Minimize total bounding box area (`W_chip × H_chip`)  
**Optimization Type:** Geometric Programming (convex in log-space)  
**Use Case:** Shape optimization, area-driven legalization

#### Key Features

- **Convex Optimization**: Guaranteed global optimum
- **Pure Shape Optimization**: Focuses on module sizes and bounding box
- **No Wirelength Term**: Only considers geometric constraints
- **Fast Convergence**: Interior-point solver converges in few iterations
- **Log-Space Formulation**: All variables and constraints in exponential form

#### Optimization Formulation

**Variables (in log-space):**
- `x_i, y_i`: Log of module center coordinates
- `w_i, h_i`: Log of module width/height
- `W_chip, H_chip`: Log of chip width/height

**Objective:**
```
minimize  W_chip · H_chip
```

**Constraints:**
1. **Area constraints**: `w_i · h_i = A_i` (module area)
2. **Aspect ratio**: `1/r_max ≤ h_i/w_i ≤ r_max`
3. **Non-overlap** (GP form): Via constraint graphs and relative positioning
4. **Boundary**: `x_i + w_i ≤ W_chip`, `y_i + h_i ≤ H_chip`

#### Command-Line Arguments

```bash
python tools/legalizer/gp_area_legalizer.py <netlist.yaml> <die.yaml> [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `netlist.yaml` | positional | required | Input netlist file |
| `die.yaml` | positional | required | Input die specification |
| `--outfile FILE` | string | None | Output result YAML file |
| `--max_ratio FLOAT` | float | 3.0 | Maximum aspect ratio (r_max) |
| `--num_iter INT` | int | 15 | Number of constraint graph rebuild iterations |
| `--verbose` | flag | False | Enable detailed solver output |
| `--relax_factor FLOAT` | float | None | Initial relaxation factor for constraints |
| `--plot` | flag | False | Generate layout visualization |

**Example Usage:**

```bash
# Minimize bounding box area
python tools/legalizer/gp_area_legalizer.py \
  netlist.yaml die.yaml \
  --outfile area_optimized.yaml \
  --max_ratio 2.5 --num_iter 20 --verbose

# With constraint relaxation (for difficult instances)
python tools/legalizer/gp_area_legalizer.py \
  netlist.yaml die.yaml \
  --outfile result.yaml \
  --relax_factor 1.5 --num_iter 30
```

---

### 2.2 `gp_wl_legalizer.py` - Wirelength Minimization via GP

**Objective:** Minimize relative stretch (convex wirelength proxy)  
**Optimization Type:** Geometric Programming (convex in log-space)  
**Use Case:** Wirelength-driven legalization, routability optimization

#### Key Features

- **Convex Wirelength Proxy**: Uses relative stretch instead of HPWL
- **Guaranteed Convergence**: Convex optimization finds global optimum
- **Net-Aware**: Explicitly models nets and bounding boxes
- **Shape + Position Optimization**: Co-optimizes module shapes and positions
- **Fast and Scalable**: Handles 100+ modules efficiently

#### Optimization Formulation

**Variables (in log-space):**
- `x_i, y_i, w_i, h_i`: Module variables (log-space)
- `X_max[k], X_min[k], Y_max[k], Y_min[k]`: Net bounding box variables

**Objective (Relative Stretch):**
```
minimize  Σ (X_max[k]/X_min[k] + Y_max[k]/Y_min[k])
          k ∈ nets
```

This is a **convex approximation** of HPWL. Minimizing stretch encourages:
- Small net bounding boxes
- Balanced span in X and Y directions
- Good routability

**Constraints:**
1. **Area constraints**: `w_i · h_i = A_i`
2. **Aspect ratio**: `1/r_max ≤ h_i/w_i ≤ r_max`
3. **Non-overlap** (GP form): Via constraint graphs
4. **Boundary**: `x_i + w_i ≤ W_chip`, `y_i + h_i ≤ H_chip`
5. **Net bounding box**: 
   - `X_max[k] ≥ x_i` for all modules i in net k
   - `X_min[k] ≤ x_i` for all modules i in net k
   - Similar for Y direction

#### Command-Line Arguments

```bash
python tools/legalizer/gp_wl_legalizer.py <netlist.yaml> <die.yaml> [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `netlist.yaml` | positional | required | Input netlist file |
| `die.yaml` | positional | required | Input die specification |
| `--outfile FILE` | string | None | Output result YAML file |
| `--max_ratio FLOAT` | float | 3.0 | Maximum aspect ratio (r_max) |
| `--num_iter INT` | int | 15 | Number of constraint graph rebuild iterations |
| `--alpha FLOAT` | float | 1.0 | LSE smoothing parameter for net bounding boxes |
| `--verbose` | flag | False | Enable detailed solver output |
| `--plot` | flag | False | Generate layout visualization |

**Example Usage:**

```bash
# Minimize wirelength (via relative stretch)
python tools/legalizer/gp_wl_legalizer.py \
  netlist.yaml die.yaml \
  --outfile wl_optimized.yaml \
  --num_iter 25 --verbose

# Fine-tune LSE smoothing
python tools/legalizer/gp_wl_legalizer.py \
  netlist.yaml die.yaml \
  --outfile result.yaml \
  --alpha 2.0 --max_ratio 2.5 --num_iter 30
```

---

## Comparison Table

| Tool | Solver | Optimization Type | Objective | Convergence | Use Case |
|------|--------|-------------------|-----------|-------------|----------|
| **legalizer.py** | GEKKO | Non-convex (MINLP) | HPWL + Overlap | Local optima | General-purpose, supports local mode |
| **glb_legalizer.py** | CasADi+Ipopt | Non-convex (NLP) | LSE-HPWL + Soft overlap | Local optima | Fast global legalization, active set |
| **gp_area_legalizer.py** | GP (log-space) | Convex | Bounding box area | Global optimum | Area minimization, shape optimization |
| **gp_wl_legalizer.py** | GP (log-space) | Convex | Relative stretch (WL proxy) | Global optimum | Wirelength minimization, routability |

### When to Use Which Tool?

**Use `legalizer.py` if:**
- You need local legalization (`--small_steps`) for fine-tuning
- You want flexible constraint modeling (expression trees)
- You have complex custom objectives

**Use `glb_legalizer.py` if:**
- You need fast global legalization (100+ modules)
- You want to use active set optimization (`--radius`)
- You need per-iteration visualization (`--plot`)
- You want to optimize rectilinear modules

**Use `gp_area_legalizer.py` if:**
- Your primary goal is minimizing chip area
- You want guaranteed global optimum
- You don't care much about wirelength initially

**Use `gp_wl_legalizer.py` if:**
- Your primary goal is minimizing wirelength
- You want guaranteed global optimum
- You need routability optimization

### Typical Multi-Stage Flow

```bash
# Stage 1: GP-based initial placement (global optimum)
python tools/legalizer/gp_wl_legalizer.py \
  netlist.yaml die.yaml --outfile gp_result.yaml

# Stage 2: NLP-based refinement (handle residual overlaps)
python tools/legalizer/glb_legalizer.py \
  gp_result.yaml die.yaml --outfile refined.yaml \
  --radius 1.0 --tau_initial 0.0001

# Stage 3: Local fine-tuning (optional)
python tools/legalizer/legalizer.py \
  refined.yaml die.yaml --outfile final.yaml \
  --small_steps --radius 0.3 --num_iter 50
```

---

## Dynamic Tau Parameter (for Non-Convex Solvers)

A key feature of `legalizer.py` and `glb_legalizer.py` is the use of a **dynamic tau (τ) parameter** for soft constraints:

**Purpose:**
- Controls the "softness" of no-overlap constraints
- Larger τ → softer constraints → more exploration
- Smaller τ → harder constraints → stricter legality

**Dynamic Decay Strategy:**
```python
τ[iteration] = τ_initial × (τ_decay ^ iteration)
```

**Benefits:**
1. **Early iterations (large τ)**: Explore global solution space, escape local minima
2. **Late iterations (small τ)**: Enforce strict legality, eliminate overlaps
3. **Balanced exploration/exploitation**: Better final quality

**Tuning Guidelines:**
- **τ_initial**: 
  - Small designs: 0.0001 - 1
  - Large designs: 1e2 - 1e6
- **τ_decay**:
  - Fast tightening: 0.3 - 0.5
  - Gradual tightening: 0.95

---

## Installation

### Requirements

- Python 3.11+
- Required packages (for all tools):
  ```bash
  pip install numpy pyyaml
  ```

- For `legalizer.py`:
  ```bash
  pip install gekko
  ```

- For `glb_legalizer.py`:
  ```bash
  pip install casadi
  ```

- For GP legalizers (`gp_area_legalizer.py`, `gp_wl_legalizer.py`):
  ```bash
  pip install casadi scipy
  ```

- For visualization (all tools):
  ```bash
  pip install matplotlib
  ```



# Verify installation
python tools/legalizer/glb_legalizer.py --help
```

---


**Format:** Each benchmark has two files:
- `*.netlist.yaml`: Module specifications and connectivity
- `*.die.yaml`: Die boundary and constraints

---

## Visualization Features

All tools support visualization via the `--plot` flag:

### `legalizer.py` Visualization
- Saves images in `example_visuals1/`
- Shows module positions and shapes per iteration
- Terminal modules marked in red

### `glb_legalizer.py` Visualization
- Saves images in specified `--plot_dir`
- Filename format: `iter_001.png`, `iter_002.png`, ...
- **Title shows**: Iteration number, HPWL, Overlap area, τ value
- **Features**:
  - Rectilinear modules (trunk + branches) with color coding
  - Trunk edges thicker than branch edges
  - Terminal modules as red dots
  - Die boundary as blue dashed rectangle
  - Grid overlay for positioning reference

**Example visualization title:**
```
Iteration 5
HPWL: 129.47, Overlap: 0.0000, τ: 6.25e-05
Die: 10.0 × 10.0, Modules: 28, Terminals: 1
```

### Creating Animations

```bash
# Generate all iterations with --plot
python tools/legalizer/glb_legalizer.py \
  netlist.yaml die.yaml --outfile result.yaml \
  --plot --plot_dir frames --num_iter 30

# Create GIF using ImageMagick
convert -delay 20 -loop 0 frames/iter_*.png optimization.gif
```

---

## Output Format

All tools output a YAML file compatible with the FRAME netlist format:

```yaml
Modules:
  module_name:
    rectangles: [[x, y, w, h], ...]  # Center-based format
    area: <float>
    # Optional fields: fixed, terminal, hard, aspect_ratio

Nets:
  - [module1, module2, ...]
  # or with weights:
  - {modules: [module1, module2], weight: 2.5}
```

---



### Rectilinear Module Support

Both `legalizer.py` and `glb_legalizer.py` support rectilinear modules:

**Netlist format:**
```yaml
module_name:
  rectangles:
    - [x, y, w, h]  # Trunk (location: TRUNK)
    - {center: [x, y], shape: [w, h], location: NORTH}
    - {center: [x, y], shape: [w, h], location: SOUTH}
    - {center: [x, y], shape: [w, h], location: EAST}
    - {center: [x, y], shape: [w, h], location: WEST}
```

**Constraints:**
1. **Attachment**: Branches must stay connected to trunk
2. **Intra-module non-overlap**: Branches don't overlap each other
3. **Inter-module non-overlap**: All rectangle pairs between modules
4. **Min Area**: All rectangles' min area
5. **Aspect_ratio**: All rectangles' aspect_ratio


### Geometric Programming Theory

**Why GP works for legalization:**

1. **Monomial/Posynomial form**: Module area, aspect ratio, and boundary constraints naturally fit GP form
2. **Log-space convexity**: Log transformation makes the problem convex
3. **Global optimality**: Convex optimization guarantees finding the global optimum
4. **Efficiency**: Interior-point methods solve GP in polynomial time

**Limitations:**
- Cannot directly model HPWL (non-posynomial)
- Need constraint graphs for non-overlap

**Solutions:**
- Use relative stretch as wirelength proxy
- Iteratively rebuild constraint graphs based on current solution
- Use GP for initial placement, NLP for refinement

---


## License & Citation

**Author:** Ylham Imam, FRAME Project  
**License:** MIT License  
**Repository:** https://github.com/jordicf/FRAME

For technical details and theoretical background, please refer to:
- Source code and inline documentation
- FRAME project documentation
- Geometric Programming literature (Boyd et al.)


---

## Important Notes

### Recommended Tool Selection

**For production use and full feature support, `legalizer.py` is the recommended choice:**

- **Most Complete Feature Set**: Full implementation with expression trees, model wrapper, and comprehensive constraint handling
- **Local Legalization Mode**: Unique `--small_steps` flag for fine-grained refinement
- **Flexible Constraint Modeling**: Expression tree framework allows for custom objectives and constraints
- **Mature and Stable**: Well-tested implementation with extensive validation
- **Comprehensive Documentation**: Full support for all module types (soft, hard, fixed, terminals)
- **Rectilinear Module Support**: Complete trunk+branches handling

### Development Status

**`glb_legalizer.py` is currently under development:**

- **In Progress**: development for simplification and deleting for strop cases.


**Use `legalizer.py` for reliable, production-ready legalization workflows.**

---

**For more examples and detailed usage, see the source code and inline documentation.**
