# `glbfloor` example 

```
frame netgen --type grid --size 2 3 -o 1.yml
frame spectral --die 2x3 -o 2.yml 1.yml
frame draw --die 2x3 2.yml
```

<img src="2.gif" alt="spectral" style="width: 200px;"/>

---

```
frame glbfloor -d 2x3 -g 4x4 -a 0.3 -i 6 --out-netlist 3-4x4-0.3-netlist.yml --out-allocation 3-4x4-0.3-alloc.yml -p 3-4x4-0.3 2.yml
```

![glbfloor-4x4-0.3-0](3-4x4-0.3-0.png)
![glbfloor-4x4-0.3-1](3-4x4-0.3-1.png)
![glbfloor-4x4-0.3-2](3-4x4-0.3-2.png)
![glbfloor-4x4-0.3-3](3-4x4-0.3-3.png)
![glbfloor-4x4-0.3-4](3-4x4-0.3-4.png)
![glbfloor-4x4-0.3-5](3-4x4-0.3-5.png)

Simple plot (no annotation nor borders):

```
frame glbfloor -d 2x3 -g 4x4 -a 0.3 -i 6 --out-netlist 3-4x4-0.3-netlist.yml --out-allocation 3-4x4-0.3-alloc.yml -p 3-4x4-0.3-simple --simple-plot 2.yml
```

![glbfloor-4x4-0.3-simple-0](3-4x4-0.3-simple-0.png)
![glbfloor-4x4-0.3-simple-1](3-4x4-0.3-simple-1.png)
![glbfloor-4x4-0.3-simple-2](3-4x4-0.3-simple-2.png)
![glbfloor-4x4-0.3-simple-3](3-4x4-0.3-simple-3.png)
![glbfloor-4x4-0.3-simple-4](3-4x4-0.3-simple-4.png)
![glbfloor-4x4-0.3-simple-5](3-4x4-0.3-simple-5.png)

```
frame draw --die 2x3 --alloc 3-4x4-0.3-alloc.yml 3-4x4-0.3-netlist.yml -o 3.gif
```

<img src="3.gif" alt="spectral" style="width: 200px;"/>

---

The folllowing results are not refined and serve to show how the initial grid and the value of alpha affect the initial allocation.

```
frame glbfloor -d 2x3 -g 4x4 -a 0.5 -p 3-4x4-0.5 2.yml
```

![glbfloor-4x4-0.5-0](3-4x4-0.5-0.png)


```
frame glbfloor -d 2x3 -g 8x8 -a 0.3 -p 3-8x8-0.3 2.yml
```

![glbfloor-8x8-0.3-0](3-8x8-0.3-0.png)

```
frame glbfloor -d 2x3 -g 8x8 -a 0.5 -p 3-8x8-0.5 2.yml
```

![glbfloor-8x8-0.5-0](3-8x8-0.5-0.png)
