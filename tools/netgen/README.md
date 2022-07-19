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
  graph TD;
      A..B;
      B..C;
      C..D;
      D..A;
```
