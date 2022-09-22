# Spectral chip planning

This tool computes an initial location for each module of the netlist using a combination of spectral
and force-directed methods.

The tool must be invoked as follows:

```
usage: frame spectral [options]

Compute the initial location for each module of the netlist using a combination of spectral and force-directed methods.

positional arguments:
  netlist               input file (netlist)

options:
  -h, --help            show this help message and exit
  -v, --verbose
  -d <WIDTH>x<HEIGHT> or FILENAME, --die <WIDTH>x<HEIGHT> or FILENAME
                        size of the die (width x height) or name of the file
  -i, --init            use initial coordinates
  --bestof BESTOF       number of floorplans generated to select the best. Default: 5
  -o OUTFILE, --outfile OUTFILE
                        output file (netlist)
```

The tool reads a `netlist` in which each module is expected to have a non-zero area. The size of the die can also
be specified in two different ways:
* `--die <widht>x<height>` (e.g., `--die 11.5x8`), indicating the width and height of the die, or
* `--die filename`, specifying the name of a file containing the description of the die.

Example:
```
frame spectral --die 32.7x46.5 -o out_net.yml in_net.yml
```

The tool defines the center of each module in the netlist. The previous values of the centers are
overwritten.

In case the option `--init` is used, the initial location of the modules is read from the input file
(instead of generating them randomly). In this case, only one floorplan is generated, i.e., the
option `--bestof` is ignored.
