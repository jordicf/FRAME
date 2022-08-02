# Block area ratios in cell grid

## Initial model

### Model definition

Here we consider a simplified floorplanning model where we have to place rectangular modules
(blocks) in a cell grid.

Each cell $c$ is characterized by four parameters: the coordinates of the center, $X_c$ and $Y_c$, and
its shape (width and height), $W_c$ and $H_c$. We define the area of the cell as $A_c = W_c H_c$.

The input of the model is thus the area of the blocks to place, the wire costs per unit length
between each pair of blocks, the number of rows and columns in the cell grid and the shape of the
cells. We will also use an $\alpha$ parameter to balance the trade-off between the total wire length
and dispersion.

#### Variables

* For each block $b$ we use the variables $x_b$ and $y_b$ to represent the centroid of the block.
* For each block $b$ we use the variables $dx_b$ and $dy_b$ to represent the dispersion of the
  block.
* For each block $b$ and each cell $c$ we use the variable $a_{bc} \in [0, 1]$ that represents the
  ratio of $A_c$ used by block $b$.

#### Constraints

* Cells cannot be over-occupied: $$\forall c: \sum_{b} a_{bc} \le 1$$
* A block must have sufficient area: $$\forall b: \sum_{c} a_{bc} \ge A_b$$
* Centroid of a block: $$\forall b: x_b = \frac{1}{A_b} \sum_{c} A_c X_c a_{bc}\qquad y_b = \frac{1}{A_b} \sum_{c} A_c Y_c a_{bc}$$
* Dispersion of a block: $$\forall b: dx_b = \sum_{c} A_c a_{bc} (x_b - X_c)^2\qquad dy_b = \sum_{c} A_c a_{bc} (y_b - Y_c)^2$$

#### Cost function

The cost function is multi-objective. It tries to minimize:

* the total wirelength:

$$
\mathrm{WL} = \sum_{e=(b_i, b_j)} c_e ((x_{b_i}-x_{b_j})^2+(y_{b_i}-y_{b_j})^2)
$$

* the total dispersion of the blocks:

$$
\mathrm{D} = \sum_{b} (dx_b + dy_b)
$$

Note that those are competing objectives. On the one hand, to reduce wirelength, the blocks get very
disperse to try to have all the centroids in the same point. On the other hand, to reduce the
dispersion of the blocks, they get compact and the centroids move farther apart. To balance the
trade-off between both objectives we use a hyperparameter $\alpha \in [0, 1]$, so the cost function
to minimize is:

$$
\alpha \mathrm{WL} + (1-\alpha) \mathrm{D}
$$

### Results over a small example

The following plots show the optimal ratios of each block of a small example using different values
of $\alpha$ ranging from 0 to 1 with steps of 0.1. We used 4 blocks of areas 1, 2, 3, and 12, and
wire costs of 1 for all the pairs. The cell grid is made up of 8 $\times$ 8 square unit cells.

<img src="results/fp-0.0.png"/>
<img src="results/fp-0.1.png"/>
<img src="results/fp-0.2.png"/>
<img src="results/fp-0.3.png"/>
<img src="results/fp-0.4.png"/>
<img src="results/fp-0.5.png"/>
<img src="results/fp-0.6.png"/>
<img src="results/fp-0.7.png"/>
<img src="results/fp-0.8.png"/>
<img src="results/fp-0.9.png"/>
<img src="results/fp-1.0.png"/>

The value of $\alpha$ coincides with the expected behavior of the competing objectives explained
before.

We observe that the total dispersion is the same with values of $\alpha$ equal to 0 and 0.1, and
that the total wire length is 0 for $\alpha$ equal or greater than 0.5.

### Discussion

* The model is non-convex, so the solution is almost surely not the global optimum. Therefore, the
  initial value of the variables is important. Spectral placement algorithms could be used to guess an
  initial location of the blocks with the hope of getting a lower local optimum.
* Using the Euclidean distance to measure the dispersion pushes the blocks to take a circular shape.
  We could try to use other distances, such as Manhattan's or Chebyshev's, but we would lose
  differentiability, although this might not be a huge problem.
* We observe that the minimization of the total dispersion tends to focus on the blocks with more
  area. This makes the model ignore the blocks with less area, and they end up having strange shapes,
  sometimes even not connected (for example, see the block with area 1 with $\alpha$ equal to 0.2).
  A better model could possibly be obtained by considering the *normalized dispersion*, which we
  define as the dispersion divided of the block divided by its area.
