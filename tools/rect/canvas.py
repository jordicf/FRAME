from PIL import Image, ImageDraw

def rgb(r,g,b):
    rs, gs, bs = format(r,'x'), format(g,'x'), format(b,'x')
    while len(rs) < 2:
        rs = "0" + rs
    while len(gs) < 2:
        gs = "0" + gs
    while len(bs) < 2:
        bs = "0" + bs
    return "#" + rs + gs + bs

def colmix(c1, c2, p):
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2
    if p <= 0:
        return rgb(r1,g1,b1)
    if p >= 1:
        return rgb(r2,g2,b2)
    r = round( r2 * p + r1 * (1 - p) )
    g = round( g2 * p + g1 * (1 - p) )
    b = round( b2 * p + b1 * (1 - p) )
    return rgb(r,g,b)

class Canvas:
    def __init__(self, width: int = 800, height = 800):
        self.x0 = 0.0
        self.y0 = 0.0
        self.x1 = float(width)
        self.y1 = float(height)
        self.width = float(width)
        self.height = float(height)
        self.canvas = Image.new("RGB", (width, height))
        self.context = ImageDraw.Draw(self.canvas)
        self.clear()
    def setcoords(self, x0: float, y0: float, x1: float, y1: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    def interpolate( self, point ):
        (x1, y1) = point
        xi0, yi0, xi1, yi1 = self.x0, self.y0, self.x1, self.y1
        xo0, yo0, xo1, yo1 = 0, 0, self.width, self.height
        x2, y2 = (x1 - xi0) / (xi1 - xi0), (y1 - yi0) / (yi1 - yi0)
        x3, y3 = x2 * (xo1 - xo0) + xo0, y2 * (yo1 - yo0) + yo0
        return (x3, y3)
    def drawbox(self, box, col="#FFFFFF", out="#000000"):
        (t1, b1) = box
        t2, b2 = self.interpolate(t1), self.interpolate(b1)
        shape = [ t2, b2 ]
        self.context.rectangle(shape, fill=col, outline=out)
    def clear(self, col="#FFFFFF"):
        shape = [ (0,0), (self.width, self.height) ]
        self.context.rectangle(shape, fill="#FFF", outline="#FFF")
    def show(self):
        self.canvas.show()
