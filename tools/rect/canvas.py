# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).


from PIL import Image, ImageDraw


def rgb(r: int, g: int, b: int) -> str:
    """
    Given three values, red green and blue, from 0 to 255, turns them into a hex code for the RGB color they represent.
    """
    rs, gs, bs = format(r, 'x'), format(g, 'x'), format(b, 'x')
    while len(rs) < 2:
        rs = "0" + rs
    while len(gs) < 2:
        gs = "0" + gs
    while len(bs) < 2:
        bs = "0" + bs
    return "#" + rs + gs + bs


def color_mix(c1: tuple[int, int, int], c2: tuple[int, int, int], p: float) -> str:
    """
    Given two colors: c1 = (R1, G1, B1) and c2 = (R2, G2, B2) and a mixing factor p, returns p*c2 + (1-p)*c1 in hex.
    """
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2
    if p <= 0:
        return rgb(r1, g1, b1)
    if p >= 1:
        return rgb(r2, g2, b2)
    r = round(r2 * p + r1 * (1 - p))
    g = round(g2 * p + g1 * (1 - p))
    b = round(b2 * p + b1 * (1 - p))
    return rgb(r, g, b)


def cross_dot_implementation(self, point: tuple[float, float], color: str, radius: float):
    width_ratio = (self.x1 - self.x0) / self.width
    height_ratio = (self.y1 - self.y0) / self.height
    (x0, y0) = point
    x1, y1 = x0 - radius * width_ratio, y0 - radius * height_ratio
    x2, y2 = x0 + radius * width_ratio, y0 + radius * height_ratio
    x3, y3 = x0 - radius * width_ratio, y0 + radius * height_ratio
    x4, y4 = x0 + radius * width_ratio, y0 - radius * height_ratio
    self._draw_simple_line(((x1, y1), (x2, y2)), color=color)
    self._draw_simple_line(((x3, y3), (x4, y4)), color=color)


def thin_cross_dot_implementation(self, point: tuple[float, float], color: str, radius: float):
    width_ratio = (self.x1 - self.x0) / self.width
    height_ratio = (self.y1 - self.y0) / self.height
    (x0, y0) = point
    x1, y1 = x0 - radius * width_ratio, y0 - radius * height_ratio
    x2, y2 = x0 + radius * width_ratio, y0 + radius * height_ratio
    x3, y3 = x0 - radius * width_ratio, y0 + radius * height_ratio
    x4, y4 = x0 + radius * width_ratio, y0 - radius * height_ratio
    self._draw_simple_line(((x1, y1), (x2, y2)), color=color, width=1)
    self._draw_simple_line(((x3, y3), (x4, y4)), color=color, width=1)


def solid_line_implementation(self, line: tuple[tuple[float, float], tuple[float, float]], color: str,
                              thickness: float):
    self._draw_simple_line(line, color=color, width=thickness)


class Canvas:
    """
    The Canvas class implements some useful functions for generating images.
    x0: The x coordinate of the upper left corner in virtual space
    y0: The y coordinate of the upper left corner in virtual space
    x1: The x coordinate of the bottom right corner in virtual space
    y1: The y coordinate of the bottom right corner in virtual space
    width: The image width
    height: The image height
    canvas: The image itself
    context: The class that implements the modifications for the image.
    """

    def __init__(self, width: int = 800, height: int = 800):
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.x1: float = float(width)
        self.y1: float = float(height)
        self.width: float = float(width)
        self.height: float = float(height)
        self.canvas: Image.Image = Image.new("RGBA", (width, height))
        self.overlay: Image.Image = Image.new("RGBA", (width, height))
        self.context: ImageDraw.ImageDraw = ImageDraw.Draw(self.overlay)
        self.clear()
        self.dot_implementations = {"cross": cross_dot_implementation,
                                    "thin_cross": thin_cross_dot_implementation}
        self.line_implementations = {"solid": solid_line_implementation}

    @staticmethod
    def hex_breakdown(col: str | None) -> tuple[int, int, int, int] | None:
        if col is None:
            return None
        if col[0] == "#":
            col = col[1:]
        if len(col) == 3:
            color = int(col[0], 16) * 16, int(col[1], 16) * 16, int(col[2], 16) * 16, 255
        elif len(col) == 4:
            color = int(col[0], 16) * 16, int(col[1], 16) * 16, int(col[2], 16) * 16, int(col[3], 16) * 16
        elif len(col) == 6:
            color = int(col[0:2], 16), int(col[2:4], 16), int(col[4:6], 16), 255
        elif len(col) == 8:
            color = int(col[0:2], 16), int(col[2:4], 16), int(col[4:6], 16), int(col[6:8], 16)
        else:
            raise Exception("Unknown hex color " + col)
        return color

    def set_coords(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """
        Tells the class the coordinates for the top left and bottom right corners
        :param x0: The x coordinate of the upper left corner in virtual space
        :param y0: The y coordinate of the upper left corner in virtual space
        :param x1: The x coordinate of the bottom right corner in virtual space
        :param y1: The y coordinate of the bottom right corner in virtual space
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def interpolate(self, point: tuple[float, float]) -> tuple[float, float]:
        """
        Moves a point from virtual space to image space
        :param point: The point (x,y) in virtual space
        """
        (x1, y1) = point
        xi0, yi0, xi1, yi1 = self.x0, self.y0, self.x1, self.y1
        xo0, yo0, xo1, yo1 = 0, 0, self.width, self.height
        x2, y2 = (x1 - xi0) / (xi1 - xi0), (y1 - yi0) / (yi1 - yi0)
        x3, y3 = x2 * (xo1 - xo0) + xo0, y2 * (yo1 - yo0) + yo0
        return x3, y3

    def _draw_simple_line(self, line: tuple[tuple[float, float], tuple[float, float]],
                          color: str = "#000000", width: int = 2) -> None:
        (b1, e1) = line
        b2, e2 = self.interpolate(b1), self.interpolate(e1)
        print(b2, e2)
        self.context.line((b2, e2), fill=color, width=width)
        self.portray_changes()

    def portray_changes(self):
        self.canvas.paste(self.overlay, (0, 0), self.overlay)
        self.context.rectangle((0, 0, self.width, self.height), fill=(0, 0, 0, 0), outline=None)

    def drawbox(self, box: tuple[tuple[float, float], tuple[float, float]],
                col: str = "#FFFFFF", out: str | None = None) -> None:
        """
        Draws a box into the image
        :param box: The coordinates of the top left and the bottom right corners of the box in virtual space
        :param col: The color of the inside of the box
        :param out: The color of the outline of the box
        """
        (t1, b1) = box
        t2, b2 = self.interpolate(t1), self.interpolate(b1)
        shape = (t2[0], t2[1], b2[0], b2[1])
        color = self.hex_breakdown(col)
        outline = self.hex_breakdown(out)
        self.context.rectangle(shape, fill=color, outline=outline)
        self.portray_changes()

    def line(self, line: tuple[tuple[float, float], tuple[float, float]], color: str = "#000000", thickness: float = 2,
             line_type: str = "solid"):
        if line_type in self.line_implementations:
            self.line_implementations[line_type](self, line, color, thickness)
        else:
            raise Exception("Line type unknown. Allowed types: " + ', '.join(list(self.line_implementations.keys())))

    def dot(self, point: tuple[float, float], color: str = "#000000", radius: float = 5,
            dot_type: str = "cross") -> None:
        if dot_type in self.dot_implementations:
            self.dot_implementations[dot_type](self, point, color, radius)
        else:
            raise Exception("Dot type unknown. Allowed types: " + ', '.join(list(self.dot_implementations.keys())))

    def clear(self, col: str = "#FFFFFF") -> None:
        """
        Clears the image
        :param col: The background color
        """
        shape = (0, 0, self.width, self.height)
        color = self.hex_breakdown(col)
        self.context.rectangle(shape, fill=color, outline=color)
        self.portray_changes()

    def show(self) -> None:
        """
        Shows the image.
        """
        self.canvas.show()
