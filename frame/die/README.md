# The Die package

The `die` package offers the class `Die` that manages the shapes and floorplanning regions on the surface of the die.

Besides the obvious `width` and `height` attributes, a `Die` also specifies a set of regions (rectangles). Four
types of regions can be distinguished:
* **Blockages**: these are regions that cannot be used during floorplanning.
* **Ground regions**: these are regions for general purpose floorplanning.
* **Specialized regions**: these are regions with dedicated resources. They are mostly meant to be used for
FPGA-oriented floorplanning. For example, these are regions that can allocate BRAMs or DSPs. LUTs can be 
floorplanned in the non-specialized (ground) regions.
* **Fixed regions**: these regions are defined by netlists and correspond to hard modules placed in fixed
locations. They cannot be used by other modules.

The specification of the die describes the blockages and specialized regions. The fixed regions must be
explicitly obtained from a netlist. Finally, the ground regions are automatically inferred by the `Die` class.

When doing floorplanning, the die regions may need to be *refined*, i.e., broken into smaller regions for
a more accurate allocation of modules. Ground and specialized regions are the *refinable* regions of
the die.

### Example

A typical use of the `die` package is as follows:

```python
# Construction of the die. If netlist is None, no fixed regions are used.
d = Die(die_yaml_file, netlist)

# The next method internally splits the refinable regions in a way that all of them have
# an aspect ratio <= 2. At least 20 refinable regions are guaranteed in the die.
# The aspect ratio of a region is greater than or equal to 1 and is computed as max(width/height, height/width)
d.split_refinable_regions(2, 20)

# Now we can obtain the set of rectangles that correspond to refinable and fixed regions
# refinable and fixed are lists of rectangles.
refinable, fixed = d.floorplanning_rectangles()

# If you prefer to obtain rectangles by type of regions, you can do:
ground = d.ground_regions
specialized = d.specialized_regions
blockages = d.blockages
fixed = d.fixed_regions
```

### How to specify a die

A die can be specified in a YAML file. Here we have a simple example.
```yaml
width: 30
height: 20
```
In this case, a rectangular die with neither blockages nor specialized regions is described.
The whole die is a *ground* region.

The next example specifies some regions.
```yaml
width: 30
height: 20
regions: [
  [18 , 29, 4, 2, '#'],
  [2, 15 , 2, 28,'BRAM'],
  [16 , 4, 6, 2, 'DSP'],
  [16 , 12, 6, 2, 'DSP']
]
```
Each rectangle is described by a tuple `[x, y, width, height, region]`, where `x` and `y` are
the coordinates of the center of the region. The regions named `#` correspond to blockages.

Notice that *fixed* regions are not specified in the die. They must be obtained from a netlist.

