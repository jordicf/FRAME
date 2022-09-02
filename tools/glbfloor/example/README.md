# `glbfloor` examples

Note that executing `glbfloor` with `--visualize` like it is done here takes a long time to execute
(several minutes)!

## Example 1

```
frame netgen --type grid --size 3 2 -o initial-netlist.yml --die 2x3 --add-centers
frame draw --die 2x3 initial-netlist.yml -o initial.gif
```

<img src="1/initial.gif" style="width: 200px;" alt="initial"/>

## Example 1.1

```
frame glbfloor --netlist initial-netlist.yml --die 2x3 -r 2 -n 16 -a 0.3 -i 10 --out-netlist 4x4-netlist.yml --out-allocation 4x4-alloc.yml --verbose --plot-name 4x4-plot --joint-plot --separated-plot --visualize
```

We could also have specified `-g 4x4` instead of `-r 2 -n 16`.

![](1/4x4-plot-separated-0.png)
![](1/4x4-plot-separated-1.png)
![](1/4x4-plot-separated-2.png)
![](1/4x4-plot-separated-3.png)
![](1/4x4-plot-separated-4.png)
![](1/4x4-plot-separated-5.png)
![](1/4x4-plot-separated-6.png)
![](1/4x4-plot-separated-7.png)
![](1/4x4-plot-separated-8.png)
![](1/4x4-plot-separated-9.png)
![](1/4x4-plot-separated-10.png)


The optimal solution (six square modules) is not found due to the initial grid form.

Full optimization animation:

|              Module-by-module               |                  Joint                  |
|:-------------------------------------------:|:---------------------------------------:|
| ![](1/4x4-plot-separated-visualization.gif) | ![](1/4x4-plot-joint-visualization.gif) |


## Example 1.2

Same as above, but with a 3x2 initial grid:

```
frame glbfloor --netlist initial-netlist.yml --die 2x3 -g 3x2 -a 0.3 -i 10 --out-netlist 3x2-netlist.yml --out-allocation 3x2-alloc.yml --verbose --plot-name 3x2-plot --joint-plot --separated-plot --visualize
```

![](1/3x2-plot-separated-0.png)
![](1/3x2-plot-separated-1.png)

Now the optimal solution is found, and in just one optimization.

Full optimization animation:

|              Module-by-module               |                  Joint                  |
|:-------------------------------------------:|:---------------------------------------:|
| ![](1/3x2-plot-separated-visualization.gif) | ![](1/3x2-plot-joint-visualization.gif) |


## Example 2

This example includes a die blockage and a fixed block.

```
frame draw --die die.yml initial-netlist.yml -o initial.gif
```

<img src="2/initial.gif" style="width: 200px;" alt="initial"/>

```
frame glbfloor --netlist initial-netlist.yml --die die.yml -r 2 -n 16 -a 0.3 -i 10 --out-netlist final-netlist.yml --out-allocation final-alloc.yml --verbose --separated-plot --joint-plot --visualize
```

![](2/plot-separated-0.png)
![](2/plot-separated-1.png)

After one optimization, the floorplan cannot be further refined so no more optimizations are needed.

Full optimization animation:

|            Module-by-module             |                Joint                |
|:---------------------------------------:|:-----------------------------------:|
| ![](2/plot-separated-visualization.gif) | ![](2/plot-joint-visualization.gif) |
