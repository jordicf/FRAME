<img src="doc/pict/Frame.png" alt="drawing" style="height: 35px;"/> 

# FRAME: Floorplanning with RectilineAr ModulEs

[![Lint & Test](https://github.com/jordicf/FRAME/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/jordicf/FRAME/actions/workflows/python-app.yml)

Chip planning is the first task in the physical design of integrated circuits.
During chip planning, the location and shapes of large modules are defined, pins are assigned, 
and the power delivery network is laid out. The quality of the chip, 
in terms of area, wirelength and timing, highly depends on the top-level
early decisions taken during chip planning.

`FRAME` is a framework for floorplanning, which is the chip planning stage
that positions and shapes the modules on the die. Modules can range from 
hard fixed-size blocks to soft free-form IP cores with several million gates
and embedded memories.

Modern floorplanning frameworks must provide features that should enable:
* A high-degree of automation with a human-in-the-loop approach.
* A friendly interaction with chip architects to specify preferences and
physical constraints.
* A diversity of non-rectangular shapes.

Automation is achieved by running efficient algorithms on mathematical models
(abstract approximations) of the problem.

If we consider _floorplanning_ as an evolutive process, a floorplan can be
represented at  different levels of abstraction. Depending on the level
of detail, a module can be seen as a point, a circle, a rectangle or a
rectilinear shape. When floorplanning is automated, a pipeline of optimization 
tasks based on mathematical models is typically envisioned, as shown in the 
figure below.

<img src="doc/pict/FPprocess.png" alt="Evolutive floorplanning" style="height: 180px;"/>

Modern floorplans require shapes beyond the conventional rectangles, e.g., L-shapes,
T-shapes, U-shapes, Z-shapes, etc. Moreover, a close interaction with the chip architect is often
required, e.g., by fixing blocks or by defining non-rectangular dies. Using a simple trick,
non-rectangular dies can be represented by including _fake_ blocks that determine the
blockages inside the rectangular die, as shown in the following figure.

<img src="doc/pict/FakeFixedBlocks.png" alt="Fake and fixed blocks" style="height: 200px;"/>

The die where the system must be floorplanned may contain dedicated regions.
A typical example is an
FPGA die with slices dedicated to BRAMs or DSPs, as shown in the figure below.

<img src="doc/pict/FPGA_structure.png" alt="Die with dedicated regions" style="height: 200px;"/>

Modules may use resources from different regions and floorplanning must take into account
where these resources are located on the die.

## The `FRAME` pipeline

Floorplanning is a multi-objective problem that cannot be solved using simple algorithmic techniques.
`FRAME` advocates for a multistep approach moving from coarse abstractions of the modules (e.g., points)
to detailed representations (e.g., rectilinear shapes). At each level, a suitable algorithmic strategy is
used that combines the accuracy of the representation with the optimality of the solution.
`FRAME` exploits mathematical and algorithmic methods to progressively refine the floorplanning infomation.

A possible pipeline of techniques for the floorplanning flow is next described.
At the beginning, modules can be treated as points or circles. Spectral methods, based on the computation 
of eigenvectors and eigenvalues from the netlist graph, can be used to find good relative positions 
of the modules. Next, force-directed methods can be used to find a good spreading across the die, possibly 
using the eigenvectors as initial solutions. Modules can then be represented as clouds that can adopt 
different shapes to accomodate the required area. Gradient-based non-linear optimization techniques can be 
used to reshape the modules using non-rectilinear free forms. The non-optimality of local minima can be
mitigated by providing a good initial approximation obtained from the previous stage.
Combinatorial optimization techniques (e.g., SAT or pseudo-Boolean optimization) can next be used to find 
rectilinear approximations of the modules on a virtual grid.
A final step may use again non-linear optimization to enforce legal locations and shapes such that
module overlaps are eliminated.

`FRAME` is designed in a modular way such that new stages can be inserted in the pipeline. These new
stages can help smoothing the jumps between different levels of abstractions and enabling the
interaction with the architect through the definition of physical constraints on the final floorplan.

