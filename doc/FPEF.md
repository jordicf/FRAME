<img src="https://github.com/jordicf/FRAME/blob/master/doc/pict/Frame.png" alt="drawing" style="height: 35px;"/>

# Floorplan Exchange Format (FPEF)
**Author:** [Jordi Cortadella](https://www.cs.upc.edu/~jordicf)

**Date:** July 2nd, 2022

---

This document describes a [`YAML`](https://en.wikipedia.org/wiki/YAML)-based exchange format, `FPEF`,
to describe rectilinear floorplans. A rectilinear floorplan consists of blocks with rectilinear shapes,
aligned with the _x_ and _y_ axes, such as the one shown in the figure.

<figure>
<p style="text-align:center">
<img src="https://www.cs.upc.edu/~jordicf/Research/FRAME/doc/RectFP.png" alt="drawing" style="height: 150px;"/>
</p>
<figcaption style="text-align:center"><b>Rectilinear floorplan</b></figcaption>
</figure>

If we consider _floorplanning_ as an evolutive process, a floorplan can be represented at different
levels of abstraction. Depending on the level of detail, a block can be seen as a point, a circle,
a rectangle or a rectilinear shape. When floorplanning is automated, a sequence of optimization tasks
based on mathematical models is typically envisioned. `FPEF` can host different levels of abstraction
for interchanging floorplaning information between different steps of the automation process,
as shown in the figure below.

<figure>
<p style="text-align:center">
<img src="https://www.cs.upc.edu/~jordicf/Research/FRAME/doc/FPprocess.png" alt="drawing" style="height: 180px;"/>
</p>
<figcaption style="text-align:center"><b>Evolutive floorplanning at different levels of abstraction</b></figcaption>
</figure>

Modern floorplans require shapes beyond the conventional rectangles, e.g., L-shapes, T-shapes or even
C-shapes. Moreover, a close interaction with the designer is often required, e.g., by fixing blocks
or by defining non-rectangular floorplanning regions. Using a simple trick, non-rectangular regions
can be represented by including _fake_ blocks that determine the blockages inside the rectangular
die, as shown in the following figure.

<figure>
<p style="text-align:center">
<img src="https://www.cs.upc.edu/~jordicf/Research/FRAME/doc/FakeFixedBlocks.png" alt="drawing" style="height: 200px;"/>
</p>
<figcaption style="text-align:center"><b>Die with blockages and fixed blocks</b></figcaption>
</figure>

The die where the system must be floorplanned may contain dedicated regions. A typical example is an
FPGA die with slices dedicated to BRAMs or DSPs, as shown in the figure below.

<figure>
<p style="text-align:center">
<img src="https://www.cs.upc.edu/~jordicf/Research/FRAME/doc/FPGA_structure.png" alt="drawing" style="height: 200px;"/>
</p>
<figcaption style="text-align:center"><b>Die with dedicated regions</b></figcaption>
</figure>

Blocks may use resources from different regions and floorplanning must take into account where these
resources are located on the die.

## FPEF top view
Using `YAML`'s nomenclature, a floorplan is a mapping with two mandatory keys: `Blocks` and `Nets`. 
Mappings and sequences admit different syntax forms in YAML. In the examples shown in this document,
only one of these forms is used but the other forms are also accepted. In mappings, the order of the keys is irrelevant. 

The description of a floorplan has a structure like the one shown in the following example:


~~~yaml
Blocks: {
  CPU: {
    # more details later
  },

  L1-Cache: {
    # ...
  }

  # more blocks ...
}

Nets: [ # A sequence of hyperedges that represent the connectivity of the blocks
  # hyperedges ...
]
~~~


## Blocks

A block is identified by a name, e.g. `CPU`, and a set of attributes.
Here is the list of possible attributes:

* `area`: defines the area of the block. Usually it is represented as a number, e.g., `area: 150`. However,
          a regionalized area can also be specified with a mapping,
          e.g., `area: {logic: 30, BRAM: 50, DSP: 15}`.
* `center`: a list of two numbers (x and y coordinates). Example: `center: [2.5, 8]`.
* `min_shape`: it represents the minimum width and height of the block. It can be represented as one number
  (applied to width and height), or a list of two numbers (min width and min height). If not specified, no
  constraints are considered for the block. Examples: `min_shape: 3` or `min_shape: [3,4]`.
* `fixed`: indicates whether the block must be in a fixed location. Example: `fixed: true`. If not specified,
           the default value is `false`.
* `rectangles`: a list of rectangles that determine the floorplan of the block. See below.

Here is an example with some attributes:
~~~yaml
  B1: {
    area: 80,
    center: [20, 15], # Center in (x,y)=(20,15)
    min_shape: [5, 4] # min width: 5, min height: 4
  },

  B2: { # Block with one fixed rectangle
    rectangles: [60, 75, 10, 12], # Center at (60, 75), width: 10, height:12
    fixed: true
  }
~~~

#### Block geometry

A rectilinear block can be represented by a set of rectangles, as shown in the figure.

<figure>
<p style="text-align:center">
<img src="https://www.cs.upc.edu/~jordicf/Research/FRAME/doc/BlockRectangles.png" alt="drawing" style="height: 150px;"/>
</p>
<figcaption style="text-align:center"><b>Rectilinear block represented as a set of rectangles</b></figcaption>
</figure>

Each rectangle is specified as a 4- or 5-element list, `[x,y,w,h,r]`, where `x` and `y` are the
coordinates of the center, and `w` and `h` are the width and the height.
The 5th element (`r`, optional) is the name of the die region the rectangle must be associated to.
If `r` is not specified, the rectangle is associated to `Ground` (default region).

The attribute `rectangles` can be either specified just one rectangle (a list) or a set of rectangles
(a list of lists).

Here we can see an example with two blocks, each one with a different type of specification.
<figure>
<p style="text-align:center">
<img src="https://www.cs.upc.edu/~jordicf/Research/FRAME/doc/TwoBlocksRectangles.png" 
alt="drawing" style="height: 85px;"/>
</p>
<figcaption style="text-align:center"><b>Two blocks represented as rectangles</b></figcaption>
</figure>

~~~yaml
  B1: {
    rectangles: [17, 2, 14, 4], # Just one rectangle associated to Ground (red in the figure)
    fixed: true # The block is fixed (the rectangle cannot be moved)
  },

  B2: {
    rectangles: [ # a block represented by two rectangles (blue in the figure)
      [5, 4, 10, 8],         # associated to Ground (default)
      [15, 6, 10, 4, 'BRAM'] # associated to BRAM
    ]
  }
~~~


Since `FPEF` can be used to represent intermediate 
steps of a floorplanning process, the description of a block does not need to be legal, e.g., 
rectangles may overlap or fall outside the bounding box of the floorplan. The rectangles must be 
interpreted as _preferred_ regions where a block would like to be located and the information can be 
used to define the initial point of a mathematical model. At the end of a complete 
floorplanning process, a legal configuration might be required. The constraints for legalization may 
be different depending on the context.

## Nets

Nets are represented as weighted sets of blocks. The weight can be interpreted as the thickness
(number of wires) of the net. Here is an example with more details on how nets are specified:
~~~yaml
Blocks: {
  # ...
}

Nets: [ # A sequence of hyperedges that represent the connectivity of the blocks
  [B1, B2],        # An edge connecting B1 and B2 with weight=1 (default)
  [B2, B3, B4, 5], # A hyperedge connecting three blocks with weight=5
  # more hyperedges ...
]
~~~

If the last element of a hyperedge is a number, it is interpreted as the weight of the hyperedge.
If the weight is not specified, the default value is 1.