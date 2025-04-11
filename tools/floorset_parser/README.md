# Floorplan Dataset Parser
This tool processes and saves individual floorplan data from .npz files to YAML file.
To process the FloorSet it needs the python libraries torch.
The `floorplan_data` dictionary loaded from the FloorSet dataset folder is expected to have the following structure:

- `area_blocks`: (n_blocks,) ndarray containing the area targets for each block. 
   * Values must be non-negative.
   * Represents the target area assigned to each block.
   * The identifier of each block is derived from its position to this array.

- `b2b_connectivity`: (b2b_edges, 3) ndarray containing block-to-block connectivity information. 
   * Each row represents an edge as [block1, block2, weight].
   * Values must be non-negative.

- `p2b_connectivity`: (p2b_edges, 3) ndarray containing pin-to-block connectivity information. 
   * Each row represents an edge as [pin, block, weight].
   * Values must be non-negative.

- `pins_pos`: (n_pins, 2) ndarray representing the (x, y) coordinates of each pin or terminal. 
   * Values must be non-negative.
   * Each row corresponds to a pin location.

- `placement_constraints`: (n_blocks, 5) ndarray containing constraints for each block. 
   * Columns indicate: [hard, fixed, multi-instantiation, cluster, boundary].
      * Hard: Can translate and rotate but not change of shape.
      * Fixed: Like Pre-placed, the shape and location are unchangeable
      * Multi-instantiation, cluster and boundary are not considered in FRAME.
   * Values must be non-negative.

- `vertex_blocks`: Can be two type of representations,
   * (n_blocks, variable_size, 2) ndarray where each entry is a list of vertices defining the polygon shape of each block. 
      * Vertices values must be non-negative. The padding value allowed is -1.
   * (n_blocks, 4) ndarray where each entry is a rectangle as [width, height, x, y] and x,y the left-lower point.

- `b_tree`: (n_blocks - 1, 3) The B*Tree representation of the floorplans (rectangular partitions + compact floorplans)
   * (Optional)

- `metrics`: (8,) ndarray containing global metrics for the floorplan. 
   * Ordered as [area, num_pins, num_total_nets, num_b2b_nets, num_p2b_nets, num_hardconstraints, b2b_weighted_wl, p2b_weighted_wl].
   * Values must be non-negative.

Validation is enforced for:
1. Non-negative values in all arrays.
2. Consistency in the sizes of the arrays (e.g., `placement_constraints` must match `n_blocks`).
3. Expected keys in the dictionary must be a subset of: 
   `['area_blocks', 'b2b_connectivity', 'p2b_connectivity', 'pins_pos', 'placement_constraints', 'vertex_blocks', 'b_tree', 'metrics']`.

The density parameter $d$ acts as a float in the range [0, 1] representing a percentage to stress the floorplan connections:
 * For each block $B_i$ we compute the total weight $W_i = \sum_{e\in b2b \bigcup p2b} e.weight$.
 * Compute the perimeter for each block $P_i$.
 * We take the block $B_k$ with highest congestion $\max_{B_i} \frac{W_i}{P_i}$
 * Then scale all net weights with the factor $\alpha$ as
 $$\alpha = \frac{d \cdot P_k}{W_k}$$


Author: Antoni Pech Alberich
Date: 2024-12-02
Version: 1.0
