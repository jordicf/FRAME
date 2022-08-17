# Netlist generation

This is a tool to generate some netlists. The usage is

```
usage: frame netgen [options]

A netlist generator.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        output file (netlist)
  --type {grid,chain,ring,star,ring-star,one-net,htree}
                        type of netlist (grid, chain, ring, star, ring-star, one-net, htree)
  --size SIZE [SIZE ...]
                        size of the netlist
  --add-centers         add module centers (only supported for grid type, and requires to specify the die)
  --add-noise [STANDARD DEVIATION]
                        (used only if --add-centers is present) adds random gaussian noise to the centers
  --seed SEED           (used only if --add-noise is present) integer number used as a seed for the random number generator
  -d <width>x<height> or filename, --die <width>x<height> or filename
                        (used only if --add-centers is present) size of the die (width x height) or name of the file
```
The figure below depicts the netlists generated when selecting one the acceptable types.
The option `size` can be followed by one or two positive integers, indicating the number of
modules in the netlist. When selecting a grid, the number of rows and columns are specified,
e.g., `--size 4 3` for four rows and three columns.

```mermaid
  graph TB 
  
      subgraph "--type chain<br>--size 4"
      direction TB
      C1 --- C2
      C2 --- C3
      C3 --- C4
      end
      
      subgraph "--type ring --size 4"
      direction TB
      R1 --- R2
      R2 --- R3
      R3 --- R4
      R4 --- R1
      end
      
      subgraph "--type star --size 4"
      S0 --- S1
      S0 --- S2
      S0 --- S3
      S0 --- S4
      end
      
      subgraph "--type grid --size 4 3"
      G00 --- G01
      G01 --- G02
      G10 --- G11
      G11 --- G12
      G20 --- G21
      G21 --- G22
      G30 --- G31
      G31 --- G32
      G00 --- G10
      G10 --- G20
      G20 --- G30
      G01 --- G11
      G11 --- G21
      G21 --- G31
      G02 --- G12
      G12 --- G22
      G22 --- G32
      end
```
