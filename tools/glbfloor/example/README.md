# `glbfloor` example 

```
frame netgen --type grid --size 3 2 -o initial.yml --die 2x3 --add-centers
frame draw --die 2x3 initial.yml
```

<img src="initial.gif" alt="initial" style="width: 200px;"/>

---

```
frame glbfloor -d 2x3 -g 4x4 -a 0.3 -i 5 --out-netlist 4x4-0.3-netlist.yml --out-allocation 4x4-0.3-alloc.yml -p 4x4-0.3 --verbose initial.yml
```

![glbfloor-4x4-0.3-0](4x4-0.3-0.png)
![glbfloor-4x4-0.3-1](4x4-0.3-1.png)
![glbfloor-4x4-0.3-2](4x4-0.3-2.png)
![glbfloor-4x4-0.3-3](4x4-0.3-3.png)
![glbfloor-4x4-0.3-4](4x4-0.3-4.png)
![glbfloor-4x4-0.3-5](4x4-0.3-5.png)

```
frame draw --die 2x3 --alloc 4x4-0.3-alloc.yml 4x4-0.3-netlist.yml -o final.gif
```

<img src="final.gif" alt="spectral" style="width: 200px;"/>
