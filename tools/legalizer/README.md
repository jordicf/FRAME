# FRAME Legalizer

## Overview

The `tools/legalizer` folder provides a powerful, extensible module legalization tool for VLSI/IC physical design, as part of the FRAME project. It is designed to transform an initial placement of modules into a legal, overlap-free, and constraint-satisfying layout, supporting both soft and hard modules, as well as terminals (I/O points). The tool features advanced optimization strategies, visualization, and benchmark validation.

---

## Dynamic Tau Parameter: Solution Space and Quality

A key feature of this legalizer is the use of a **dynamic tau parameter** for soft constraints, which plays a crucial role in controlling the "softness" or "hardness" of overlap and proximity constraints during optimization:

- **Tau as a Soft Constraint Smoothing Factor**: The tau parameter determines how strictly the no-overlap and proximity constraints are enforced. A larger tau allows more flexibility (softer constraints), enabling the optimizer to explore a broader solution space and escape poor local minima. As tau decreases, the constraints become sharper, guiding the solution toward strict legality.
- **Dynamic Decay**: Tau is typically decayed (reduced) over iterations, starting with a high value to encourage global exploration and gradually tightening to enforce strict legality and high-quality packing.
- **Impact on Solution Quality**: Proper tuning of the initial tau and its decay rate can significantly affect the final layout quality. Too high a tau for too long may result in residual overlaps; too low a tau too early may trap the solution in suboptimal local minima. The dynamic schedule balances exploration and exploitation, leading to better wirelength, lower overlap, and more robust convergence.
- **User Control**: You can control tau via `--tau_initial` and `--tau_decay` command-line options for advanced tuning.

---

## Main Features

- **Module Legalization**: Adjusts module positions and shapes to eliminate overlaps and satisfy aspect ratio and boundary constraints.
- **Support for Multiple Module Types**: Handles soft modules (resizable), hard modules (fixed size), and terminals (fixed I/O points) in a unified framework.
- **Advanced Optimization Objectives**: Minimizes wirelength, overlap area, and supports customizable objective weights.
- **Adaptive Tolerance and Multi-Stage Optimization**: Features temperature decay, adaptive tolerances, and multi-phase optimization for robust convergence.
- **Visualization**: Generates layout images for each iteration and plots HPWL/Overlap curves.
- **Benchmark Validation**: Includes standard MCNC and GSRC benchmark datasets for testing and comparison.

---

## File Structure

- **legalizer.py**  
  Main entry point. Handles command-line parsing, data loading, model construction, optimization flow, result output, and visualization.  
  Key classes/functions: `main`, `Model`, `parse_options`, `compute_options`.

- **expr_tree.py**  
  Core for expression tree and constraint modeling. Implements symbolic expressions (`ExpressionTree`), constraints (`Equation`), and comparison operators (`Cmp`), with automatic differentiation and Gekko integration.  
  Feature: Enables complex objectives and constraints to be built and evaluated symbolically.

- **modelwrap.py**  
  Wrapper for the Gekko optimizer. Manages variables, constraints, and objectives, and provides high-level interfaces such as `build_model`, `solve`, and `verify`.  
  Feature: Supports dynamic constraint adjustment, variable fixing, and differential objectives.

- **bench-exam/**  
  Standard benchmark datasets for validation and testing.  
  - `MCNC/`: Classic circuits (ami33, ami49, etc.) in netlist/die YAML format.  
  - `GSRC/`: Scalable testcases (n10~n300) for large-scale experiments.

---

## Installation

1. **Install Python 3.10+** (recommended: use a virtual environment)
2. **Install dependencies:**

```bash
pip install gekko matplotlib
# Plus FRAME project dependencies as required
```

---

## Usage

### Command-Line Example

```bash
python tools/legalizer/legalizer.py <netlist.yaml> <die.yaml> [options]
```

**Key options:**

- `--max_ratio <float>`: Maximum aspect ratio for rectangles (default: 3.0)
- `--num_iter <int>`: Number of optimization iterations (default: 15)
- `--radius <float>`: No-overlap constraint radius and  movemet constraint (default: 1)
- `--wl_mult <float>`: Wirelength objective multiplier (default: 1)
- `--plot`: Enable per-iteration visualization
- `--small_steps`: Use small movement steps (finer legalization)
- `--verbose`: Show detailed debug information
- `--outfile <file>`: Output result YAML file
- Advanced: `--tau_initial`, `--tau_decay`, `--otol_initial`, `--otol_final`, `--rtol_initial`, `--rtol_final`, `--tol_decay` (see code for details)
- Future `--objective`: can select type of objective function, e.g., LSE(HPWL) and Quadratic Model

**Example:**

```bash
python tools/legalizer/legalizer.py bench-exam/MCNC/ami33.netlist.yaml bench-exam/MCNC/ami33.die.yaml --plot --num_iter 20 --outfile result.yaml
```

### Output

- The result is written to the specified YAML file, compatible with the input netlist format.
- If `--plot` is enabled, layout images for each iteration are saved in `example_visuals/`.
- After execution, HPWL and Overlap curves are plotted automatically.

---

## Highlights

- **Terminal-aware and mixed-type legalization**
- **Automatic tolerance adjustment and multi-stage optimization**
- **Integrated visualization and benchmark validation**
- **Highly modular and extensible for research and engineering**
- **Powered by Gekko: supports nonlinear and mixed-integer optimization**

---

## Benchmark/Test Data

- Place your own netlist/die YAML files in `bench-exam/` to use custom benchmarks.
- The tool supports standard MCNC and GSRC datasets for fair comparison and validation.

---

## License & Contact

- Author: Ylham Imam, FRAME Project
- License: MIT License
- See [LICENSE.txt](https://github.com/jordicf/FRAME/blob/master/LICENSE.txt) for details.

For technical details, please refer to the source code and inline comments. 