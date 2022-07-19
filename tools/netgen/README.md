# Netlist generation

This is a tool to generate some netlists. The usage is

```
usage: frame netgen [options]

A netlist generator

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        output file (netlist)
  --type {grid,chain,ring,star}
                        type of netlist (grid, chain, ring, star)
  --size SIZE [SIZE ...]
                        size of the netlist
```
The figure below depicts the netlists generated when selecting one the acceptable types.
The option `size` can be followed by one or two positive integers, indicating the number of
modules in the netlist. When selecting a grid, the number of rows and columns are specified,
e.g., `--size 4 3` for four rows and three columns.

```mermaid
  graph TB 
  
      subgraph "--type chain\n--size 4"
      direction TB
      C1 --- C2
      C2 --- C3
      C3 --- C4
      end
      
      subgraph ring
      direction TB
      R1 --- R2
      R2 --- R3
      R3 --- R4
      R4 --- R1
      end
      
      subgraph star
      S0 --- S1
      S0 --- S2
      S0 --- S3
      S0 --- S4
      end
      
      subgraph grid
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
