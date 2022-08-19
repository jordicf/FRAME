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


def colmix(c1: tuple[int, int, int], c2: tuple[int, int, int], p: float) -> str:
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
        self.canvas: Image.Image = Image.new("RGB", (width, height))
        self.context: ImageDraw.ImageDraw = ImageDraw.Draw(self.canvas)
        self.clear()

    def setcoords(self, x0: float, y0: float, x1: float, y1: float) -> None:
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

    def drawbox(self, box: tuple[tuple[float, float], tuple[float, float]],
                col: str = "#FFFFFF", out: str = "#000000") -> None:
        """
        Draws a box into the image
        :param box: The coordinates of the top left and the bottom right corners of the box in virtual space
        :param col: The color of the inside of the box
        :param out: The color of the outline of the box
        """
        (t1, b1) = box
        t2, b2 = self.interpolate(t1), self.interpolate(b1)
        shape = (t2[0], t2[1], b2[0], b2[1])
        self.context.rectangle(shape, fill=col, outline=out)

    def clear(self, col: str = "#FFFFFF") -> None:
        """
        Clears the image
        :param col: The background color
        """
        shape = (0, 0, self.width, self.height)
        self.context.rectangle(shape, fill=col, outline=col)

    def show(self) -> None:
        """
        Shows the image.
        """
        self.canvas.show()
