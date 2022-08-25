# Orthogonal polygons

`FRAME` can deal with rectilinear modules that are represented by
[Rectilinear Polygons](https://en.wikipedia.org/wiki/Rectilinear_polygon), 
also called Orthogonal Polygons or *Orthogons*.

We are interested in *simple* orthogons, i.e., orthogons without holes. This is a simple orthogon:

<img src="pict/no-stog.png" alt="Rectilinear floorplan" style="height: 80px;"/>

An interesting property of orthogons is that they can be partitioned into a set of disjoint rectangles.
We are interested in a particular subclass of orthogons that we call *Single-Trunk Orthogons* (STOG).

## STOG

A STOG is an orthogon that can be decomposed into a set of disjoint rectangles with the following property:
such that one of them is
called the *trunk* and the others are called *branches*, with the following property:
* One rectangle is called the trunk and the others are called the branches
* Each branch fully shares one of its edges with the trunk.

The following picture shows an orthogon that is not a STOG. Assuming that the blue rectangle is the trunk, two
of the other rectangles (left and right) are branches of the trunk, since they fully share one of the edges
with the trunk. However, the bottom rectangle is not a branch, since the common edge is not fully shared
with the trunk.

<img src="pict/trunk_branch.png" alt="Rectilinear floorplan" style="height: 120px;"/>

### $k$-STOG

A $k$-STOG is a STOG with one trunk and $k$ branches. Orthogonal rectangles are $0$-STOGs since they only
have one trunk and 0 branches, as shown in this picture:

<img src="pict/zero-stog.png" alt="Rectilinear floorplan" style="height: 80px;"/>

$1$-STOGs include all the L- and T-shaped orthogons. Here are two examples:

<img src="pict/one-stog.png" alt="Rectilinear floorplan" style="height: 80px;"/>

$2$-STOGs offer a rich variety of orthogons, as shown here:

<img src="pict/two-stog.png" alt="Rectilinear floorplan" style="height: 80px;"/>

In case you have curiosity of knowing the associated partition of rectangles, the following picture
shows the trunks (T) and branches of each case. The NSEW labels indicate the trunk edge adjacent to each branch.

<img src="pict/two-stog-rectangles.png" alt="Rectilinear floorplan" style="height: 80px;"/>

### The structure of a STOG

The structure of a STOG is characterized by the relative location of its branches, e.g., $0$-STOGs can only have one
structure (orthogonal rectangles), $1$-STOGs can have 4 different structures depending on the location of the branch
(*N*, *S*, *E*, *W*), $2$-STOGs can have 10 different structures (*NN*, *NS*, *NE*, *NW*, *SS*, *SE*, ...). In general,
$k$-STOGs can have up to $\binom{k+3}{k}$ structures.

### Why STOGs?

Some of the `FRAME` stages are based on non-convex optimization models, which typically use gradient-descent algorithms
for finding local minima. It is convenient that the constraints of these models can be represented
by differentiable functions. 


