# A drawing tool for rectilinear floorplans

This is a drawing tool for rectilinear floorplans. The tool reads a netlist in which each module can
be represented by a set of rectangles or by a center and an area. If no rectangles are specified, the module is drawn as
a circle.

## Usage

```
usage: frame draw [options]

A floorplan drawing tool

positional arguments:
  netlist               input file (netlist)

options:
  -h, --help            show this help message and exit
  --die <width>x<height> or filename
                        Size of the die (width x height) or name of the file
  -o OUTFILE, --outfile OUTFILE
                        output file (gif)
  --width WIDTH         width of the picture (in pixels)
  --height HEIGHT       height of the picture (in pixels)
  --frame FRAME         frame around the die (in pixels). Default: 40
  --fontsize FONTSIZE   text font size. Default: 20
```

The option `--die` is used to specify the size of the die. It can be done in two ways:

* By specifiying the width and the height in a string, e.g., `--die 50x40`
* By specifying the name of a file that describes the die, e.g., `--die die.yml`.
  In this case, the width and height of the die is obtained from the file.

The option `--outfile` is optional. If not specified, the name of the input file is used and a `.gif`
suffix is added (substituting the original suffix).

The `--frame` option is used to specify a frame around the die.


