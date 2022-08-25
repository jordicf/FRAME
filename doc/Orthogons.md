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
* One rectangle is called the trunk and the others are called the branches
* Each branch is adjacent to the trunk and fully shares one of its edges with the trunk

The following picture shows an orthogon that is not a STOG. Assuming that the blue rectangle is the trunk, two
of the other rectangles (left and right) are branches of the trunk, since they fully share one of the edges
with the trunk. However, the bottom rectangle is not a branch, since the common edge is not fully shared
with the trunk.

<img src="pict/trunk_branch.png" alt="Trunk and branches" style="height: 120px;"/>

### *k*-STOGs

A *k*-STOG is a STOG with one trunk and *k* branches. Orthogonal rectangles are *0*-STOGs since they only
have one trunk and 0 branches, as shown in this picture:

<img src="pict/zero-stog.png" alt="0-STOG" style="height: 80px;"/>

*1*-STOGs include all the L- and T-shaped orthogons. Here are two examples:

<img src="pict/one-stog.png" alt="1-STOGs" style="height: 80px;"/>

*2*-STOGs offer a rich variety of orthogons, as shown here:

<img src="pict/two-stog.png" alt="2-STOGs" style="height: 80px;"/>

In case you have curiosity of knowing the associated partition of rectangles, the following picture
shows the trunks (T) and branches of each case. The NSEW labels indicate the trunk edge adjacent to each branch.

<img src="pict/two-stog-rectangles.png" alt="Rectangles of 2-STOGs" style="height: 80px;"/>

### The structure of a STOG

The structure of a STOG is characterized by the relative location of its branches, e.g., *0*-STOGs can only have one
structure (orthogonal rectangles), *1*-STOGs can have 4 different structures depending on the location of the branch
(*N*, *S*, *E*, *W*), *2*-STOGs can have 10 different structures (*NN*, *NS*, *NE*, *NW*, *SS*, *SE*, ...). In general,
the number of possible structures of a *k*-STOG is (subsets with repetitions):

$$\binom{k+3}{k}$$

### Why STOGs?

Some of the `FRAME` stages are based on non-convex optimization models, which typically use gradient-descent algorithms
for finding local minima. It is convenient that the constraints used to characterize the structure of a STOG
can be modeled by differentiable functions. 

#### Example

Let us assume that a rectangle is represented by the coordinates of its center *(x,y)*, width (*w*) and height
(*h*). Let us consider a STOG with a trunk (*T*) and an *East*-branch (*B*). Then, the relative position of the branch
with regard to the trunk can be modeled with three linear constraints (one equality and two inequalities), as follows:

$$
\begin{eqnarray*}
x_B & = & x_T + \frac{w_T + w_B}{2} \\
y_B & \leq & y_T + \frac{h_T-h_B}{2} \\ 
y_B & \geq &  y_T + \frac{h_B-h_T}{2}
\end{eqnarray*}
$$

In case two branches share the same edge, a simple linear constraint may be used to avoid their overlap by
assuming an ordering on the edge, e.g.,

$$y_{B_1}-y_{B_2} \geq \frac{h_{B_2}+h_{B_1}}{2}$$

**Note:** the fact that STOGs can be modeled with linear constraints does not mean that the complete model is linear.
Unfortunately, some other constraints of the same model may not be linear.


