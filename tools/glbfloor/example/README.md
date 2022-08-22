# `glbfloor` examples

Note that executing `glbfloor` with `--visualize` like it is done here takes a long time to execute
(several minutes)!

## Example 1

```
frame netgen --type grid --size 3 2 -o initial-netlist.yml --die 2x3 --add-centers
frame draw --die 2x3 initial-netlist.yml -o initial.gif
```

<img src="1/initial.gif" alt="glbfloor-1-initial" style="width: 200px;"/>

## Example 1.1

```
frame glbfloor --netlist initial-netlist.yml --die 2x3 -r 2 -n 16 -a 0.3 -i 10 --out-netlist 4x4-netlist.yml --out-allocation 4x4-alloc.yml -p 4x4-plot --verbose --visualize
```

We could also have specified `-g 4x4` instead of `-r 2 -n 16`.

![glbfloor-1.1-plot-0](1/4x4-plot-0.png)
![glbfloor-1.1-plot-1](1/4x4-plot-1.png)
![glbfloor-1.1-plot-2](1/4x4-plot-2.png)
![glbfloor-1.1-plot-3](1/4x4-plot-3.png)
![glbfloor-1.1-plot-4](1/4x4-plot-4.png)
![glbfloor-1.1-plot-5](1/4x4-plot-5.png)
![glbfloor-1.1-plot-6](1/4x4-plot-6.png)
![glbfloor-1.1-plot-7](1/4x4-plot-7.png)
![glbfloor-1.1-plot-8](1/4x4-plot-8.png)
![glbfloor-1.1-plot-9](1/4x4-plot-9.png)
![glbfloor-1.1-plot-10](1/4x4-plot-10.png)

The optimal solution (six square modules) is not found due to the initial grid form.

Full optimization animation:

![glbfloor-1-animation](1/4x4-plot.gif)

```
frame draw --die 2x3 --alloc 4x4-alloc.yml 4x4-netlist.yml -o 4x4-joint-plot.gif
```

<img src="1/4x4-joint-plot.gif" alt="glbfloor-1.1-final" style="width: 200px;"/>


## Example 1.2

Same as above, but with a 3 x 2 initial grid:

```
frame glbfloor --netlist initial-netlist.yml --die 2x3 -g 3x2 -a 0.3 -i 10 --out-netlist 3x2-netlist.yml --out-allocation 3x2-alloc.yml -p 3x2-plot --verbose --visualize
```

![glbfloor-1.2-plot-0](1/3x2-plot-0.png)
![glbfloor-1.2-plot-1](1/3x2-plot-1.png)

Now the optimal solution is found, and in just one optimization.

Full optimization animation:

![glbfloor-1-animation](1/3x2-plot.gif)

```
frame draw --die 2x3 --alloc 3x2-alloc.yml 3x2-netlist.yml -o 3x2-joint-plot.gif
```

<img src="1/3x2-joint-plot.gif" alt="glbfloor-1.2-final" style="width: 200px;"/>


## Example 2

This example includes a die blockage and a fixed block.

```
frame draw --die die.yml initial-netlist.yml -o initial.gif
```

<img src="2/initial.gif" alt="glbfloor-2-initial" style="width: 200px;"/>

```
frame glbfloor --netlist initial-netlist.yml --die die.yml -r 2 -n 16 -a 0.3 -i 10 --out-netlist final-netlist.yml --out-allocation final-alloc.yml -p plot --verbose --visualize
```

![glbfloor-2-plot-0](2/plot-0.png)
![glbfloor-2-plot-1](2/plot-1.png)

After one optimization, the floorplan cannot be further refined so no more optimizations are needed.

Full optimization animation:

![glbfloor-2-animation](2/plot.gif)

```
frame draw --die die.yml --alloc final-alloc.yml final-netlist.yml -o final.gif
```

<img src="2/final.gif" alt="glbfloor-2-final" style="width: 200px;"/>
