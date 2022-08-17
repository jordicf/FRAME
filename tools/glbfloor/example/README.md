# `glbfloor` example 

```
frame netgen --type grid --size 3 2 -o initial.yml --die 2x3 --add-centers
frame draw --die 2x3 initial.yml
```

<img src="initial.gif" alt="initial" style="width: 200px;"/>

---

```
frame glbfloor -d 2x3 -g 4x4 -a 0.3 -i 10 --out-netlist 4x4-0.3-netlist.yml --out-allocation 4x4-0.3-alloc.yml -p 4x4-0.3 --verbose initial.yml  
```

![glbfloor-4x4-0.3-0](4x4-0.3-0.png)
![glbfloor-4x4-0.3-1](4x4-0.3-1.png)
![glbfloor-4x4-0.3-2](4x4-0.3-2.png)
![glbfloor-4x4-0.3-3](4x4-0.3-3.png)
![glbfloor-4x4-0.3-4](4x4-0.3-4.png)
![glbfloor-4x4-0.3-5](4x4-0.3-5.png)
![glbfloor-4x4-0.3-6](4x4-0.3-6.png)
![glbfloor-4x4-0.3-7](4x4-0.3-7.png)
![glbfloor-4x4-0.3-8](4x4-0.3-8.png)
![glbfloor-4x4-0.3-9](4x4-0.3-9.png)
![glbfloor-4x4-0.3-10](4x4-0.3-10.png)

```
frame draw --die 2x3 --alloc 4x4-0.3-alloc.yml 4x4-0.3-netlist.yml -o 4x4-0.3-final.gif
```

<img src="4x4-0.3-final.gif" alt="4x4-0.3-final" style="width: 200px;"/>

---

```
frame glbfloor -d 2x3 -g 3x2 -a 0.3 -i 10 --out-netlist 3x2-0.3-netlist.yml --out-allocation 3x2-0.3-alloc.yml -p 3x2-0.3 --verbose initial.yml
```

![glbfloor-3x2-0.3-0](3x2-0.3-0.png)
![glbfloor-3x2-0.3-1](3x2-0.3-1.png)

```
frame draw --die 2x3 --alloc 3x2-0.3-alloc.yml 3x2-0.3-netlist.yml -o 3x2-0.3-final.gif
```

<img src="3x2-0.3-final.gif" alt="3x2-0.3-final" style="width: 200px;"/>
