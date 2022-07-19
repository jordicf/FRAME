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

```mermaid
  graph TD
      subgraph
      A---B
      B---C
      C---D
      D---A
      end
      
      subgraph
      W---X
      X---Y
      Y---Z
      end
      
      subgraph
      M00 --- M01
      M01 --- M02
      M10 --- M11
      M11 --- M12
      M20 --- M21
      M21 --- M22
      M30 --- M31
      M31 --- M32
      M00 --- M10
      M10 --- M20
      M20 --- M30
      M01 --- M11
      M11 --- M21
      M21 --- M31
      M02 --- M12
      M12 --- M22
      M22 --- M32
      end
```
