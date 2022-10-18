# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).
from math import sqrt

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
    ratio = min(height_ratio, width_ratio)
    (x0, y0) = point
    x1, y1 = x0 - radius * width_ratio, y0 - radius * ratio
    x2, y2 = x0 + radius * width_ratio, y0 + radius * ratio
    x3, y3 = x0 - radius * width_ratio, y0 + radius * ratio
    x4, y4 = x0 + radius * width_ratio, y0 - radius * ratio
    self._draw_simple_line(((x1, y1), (x2, y2)), color=color)
    self._draw_simple_line(((x3, y3), (x4, y4)), color=color)


def thin_cross_dot_implementation(self, point: tuple[float, float], color: str, radius: float):
    width_ratio = (self.x1 - self.x0) / self.width
    height_ratio = (self.y1 - self.y0) / self.height
    ratio = min(height_ratio, width_ratio)
    (x0, y0) = point
    x1, y1 = x0 - radius * width_ratio, y0 - radius * ratio
    x2, y2 = x0 + radius * width_ratio, y0 + radius * ratio
    x3, y3 = x0 - radius * width_ratio, y0 + radius * ratio
    x4, y4 = x0 + radius * width_ratio, y0 - radius * ratio
    self._draw_simple_line(((x1, y1), (x2, y2)), color=color, width=1)
    self._draw_simple_line(((x3, y3), (x4, y4)), color=color, width=1)


def thin_circle_dot_implementation(self, point: tuple[float, float], color: str, radius: float):
    width_ratio = (self.x1 - self.x0) / self.width
    height_ratio = (self.y1 - self.y0) / self.height
    self._draw_ellipse(point, (radius * width_ratio, radius * height_ratio), color, None, 1)


def solid_circle_dot_implementation(self, point: tuple[float, float], color: str, radius: float):
    width_ratio = (self.x1 - self.x0) / self.width
    height_ratio = (self.y1 - self.y0) / self.height
    self._draw_ellipse(point, (radius * width_ratio, radius * height_ratio), None, color, 1)


def solid_line_implementation(self, line: tuple[tuple[float, float], tuple[float, float]], color: str,
                              thickness: float) -> None:
    self._draw_simple_line(line, color=color, width=thickness)


def dashed_line_implementation(self, line: tuple[tuple[float, float], tuple[float, float]], color: str,
                               thickness: float) -> None:
    ((x1, y1), (x2, y2)) = line
    # If the line is just a point, don't bother.
    if x1 == y1 and x2 == y2:
        return

    # Cohen-Sutherland Algorithm for line clipping
    min_x = min(self.x0, self.x1)
    max_x = max(self.x0, self.x1)
    min_y = min(self.y0, self.y1)
    max_y = max(self.y0, self.y1)
    inside, left, right, bottom, top = 0, 1, 2, 4, 8

    def compute_out_code(x_coord: float, y_coord: float):
        code = inside
        if x_coord < min_x:
            code |= left
        if x_coord > max_x:
            code |= right
        if y_coord < min_y:
            code |= bottom
        if y_coord > max_y:
            code |= top
        return code

    out_code1 = compute_out_code(x1, y1)
    out_code2 = compute_out_code(x2, y2)
    while True:
        if out_code1 == inside and out_code2 == inside:
            # Both points are inside the drawing region
            break
        elif out_code1 & out_code2 > 0:
            # Both points can be found on the same side outside
            return
        else:
            out_code_out = max(out_code1, out_code2)
            if out_code_out & top > 0:
                x = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1)
                y = max_y
            elif out_code_out & bottom > 0:
                x = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1)
                y = min_y
            elif out_code_out & right > 0:
                y = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1)
                x = max_x
            else:
                y = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1)
                x = min_x

            if out_code_out == out_code1:
                x1 = x
                y1 = y
                out_code1 = compute_out_code(x1, y1)
            else:
                x2 = x
                y2 = x
                out_code2 = compute_out_code(x2, y2)

    magnitude = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if magnitude == 0:
        return

    # Draw dashed line
    pos = 0.0
    inc = (thickness * 0.1) / magnitude
    while pos < 1:
        next_pos = min(1.0, pos + inc)
        point1 = (x1 + (x2 - x1) * pos, y1 + (y2 - y1) * pos)
        point2 = (x1 + (x2 - x1) * next_pos, y1 + (y2 - y1) * next_pos)
        self._draw_simple_line((point1, point2), color=color, width=thickness)
        pos = min(1.0, next_pos + 2 * inc)


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
                                    "thin_cross": thin_cross_dot_implementation,
                                    "thin_circle": thin_circle_dot_implementation,
                                    "solid_circle": solid_circle_dot_implementation}
        self.line_implementations = {"solid": solid_line_implementation,
                                     "dashed": dashed_line_implementation}

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
        self.context.line((b2, e2), fill=color, width=width)
        self.portray_changes()

    def _draw_ellipse(self, center: tuple[float, float], radii: tuple[float, float],
                      outline: str = "#000000", fill: str = "#FFFFFF", width: int = 1) -> None:
        width_ratio = (self.x1 - self.x0) / self.width
        height_ratio = (self.y1 - self.y0) / self.height
        c, r = self.interpolate(center), (radii[0] / width_ratio, radii[1] / height_ratio)
        self.context.ellipse([(c[0] - r[0], c[1] - r[1]), (c[0] + r[0], c[1] + r[1])],
                             fill=fill, outline=outline, width=width)
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

    def save(self, path: str) -> None:
        """
        Saves the image.
        """
        self.canvas.save(path)
