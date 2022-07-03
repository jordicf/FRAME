import yaml
from typing import List, Dict, Set, Deque, NamedTuple, Optional
from fp_types import valid_identifier, is_number, string_is_number, Point, Shape
from fp_keywords import KW_WIDTH, KW_HEIGHT, KW_RECTANGLES, KW_CENTER, KW_SHAPE, KW_REGION, GROUND_REGION
from fp_rectangle import Rectangle
from collections import deque


def string_layout(layout: str) -> Optional[Shape]:
    """
    Parses the layout string of the form <width>x<height> or from a filename
    :param layout: the layout string or filename
    :return: a shape if it has the form <width>x<height>, or None otherwise
    """
    numbers = layout.rsplit('x')
    if len(numbers) == 2 and string_is_number(numbers[0]) and string_is_number(numbers[1]):
        w, h = float(numbers[0]), float(numbers[1])
        assert w > 0 and h > 0, "The width and height of the layout must be positive"
        return Shape(w, h)
    return None


class GroundRegion(NamedTuple):
    rmin: int
    rmax: int
    cmin: int
    cmax: int
    area: float
    ratio: float

    def __str__(self) -> str:
        return f'<rows=({self.rmin}-{self.rmax}), cols=({self.cmin}-{self.cmax}), area={self.area}, ratio={self.ratio}>'

    def __hash__(self) -> int:
        return hash(37 * self.rmin + 13 * self.rmax + 7 * self.cmin + 23 * self.cmax)


class Layout:
    """
    Class to represent the layout of the floorplan (ground and tagged rectangles)
    """
    _width: float  # Width of the layout
    _height: float  # Height of the layout
    _regions: list[Rectangle]  # List of non-ground regions
    _ground_regions: list[Rectangle]  # List of ground regions
    _epsilon: float  # Precision when dealing with coordinates
    _x: list[float]  # List of x coordinates of potential rectangles
    _y: list[float]  # List of y coordinates of potential rectangles
    _cells: [list[list[bool]]]  # Matrix of rectangles (True occupied, False available)

    def __init__(self, filename_or_string: str):
        """
        Constructor of a layout
        :param filename_or_string: name of the YAML file (or string <width>x<height>)
        """
        shape: Shape | None = string_layout(filename_or_string)
        if shape is None:
            with open(filename_or_string) as f:
                self._parse_layout(yaml.safe_load(f.read()))
        else:
            self._parse_layout({KW_WIDTH: shape.w, KW_HEIGHT: shape.h})
        self._epsilon = min(self._width, self._height) * 10e-12
        self._calculate_region_points()
        self._calculate_cell_matrix()
        self._calculate_ground_rectangles()

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    def _parse_layout(self, tree: Dict):
        """
        Parses the regions of the layout
        :param tree: YAML tree
        """
        assert isinstance(tree, dict), "The layout is not a dictionary"
        for key in tree:
            assert key in [KW_WIDTH, KW_HEIGHT, KW_RECTANGLES], f"Unknown keyword in layout: {key}"

        assert KW_WIDTH in tree and KW_HEIGHT in tree, "Wrong format of the layout: Missing width or height"
        self._width, self._height = tree[KW_WIDTH], tree[KW_HEIGHT]
        assert is_number(self._width) and self._width > 0, "Wrong specification of the layout width"
        assert is_number(self._height) and self._height > 0, "Wrong specification of the layout height"

        self._regions = []
        if KW_RECTANGLES in tree:
            rlist = tree[KW_RECTANGLES]
            assert isinstance(rlist, list) and len(rlist) > 0, f"Incorrect specification of layout rectangles"
            if is_number(rlist[0]):
                rlist = [rlist]  # List with only one rectangle
            for r in rlist:
                self._parse_layout_rectangle(r)

    def _parse_layout_rectangle(self, r: List):
        """
        Parses a rectangle
        :param r: a YAML description of the rectangle (a list with 5 values).
        """
        assert isinstance(r, list) and len(r) == 5, "Incorrect format of layout rectangle"
        for i in range(4):
            assert is_number(r[i]) and r[i] >= 0, "Incorrect value for layout rectangle"
        assert isinstance(r[4], str) and valid_identifier(r[4])
        kwargs = {KW_CENTER: Point(r[0], r[1]), KW_SHAPE: Shape(r[2], r[3]), KW_REGION: r[4]}
        self._regions.append(Rectangle(**kwargs))

    def _calculate_region_points(self):
        """
        Calculates the list of points to be candidates for rectangle corners in the ground.
        """
        x, y = [0], [0]
        for r in self._regions:
            bb = r.bounding_box
            x.append(bb[0].x)
            x.append(bb[1].x)
            y.append(bb[0].y)
            y.append(bb[1].y)
        x.append(self._width)
        y.append(self._height)
        x.sort()
        y.sort()
        # Remove duplicates
        self._x = []
        for i, val in enumerate(x):
            if i == 0 or val > self._x[-1] + self._epsilon:
                self._x.append(float(val))
        self._y = []
        for i, val in enumerate(y):
            if i == 0 or val > self._y[-1] + self._epsilon:
                self._y.append(float(val))

    def _calculate_cell_matrix(self):
        """
        Calculates the matrix of cells. It indicates which cells are occupied by regions
        """
        self._cells = [[False for x in range(len(self._x) - 1)] for y in range(len(self._y) - 1)]
        for i in range(len(self._x) - 1):
            x = (self._x[i] + self._x[i + 1]) / 2
            for j in range(len(self._y) - 1):
                p = Point(x, (self._y[j] + self._y[j + 1]) / 2)
                for r in self._regions:
                    if r.inside(p):
                        self._cells[j][i] = True

    def _calculate_ground_rectangles(self):
        self._ground_regions = []
        g_rect = self._find_largest_ground_region()
        while g_rect is not None:
            self._ground_regions.append(g_rect)
            g_rect = self._find_largest_ground_region()

    def _find_largest_ground_region(self) -> Optional[Rectangle]:
        """
        Calculates the largest non-occupied rectangular region of the layout
        :return: the largest region
        """
        all_regions: Set[GroundRegion] = set()  # Set of all rectangular regions
        for r in range(len(self._cells)):
            height = self._y[r + 1] - self._y[r]
            for c in range(len(self._cells[r])):
                if not self._cells[r][c]:
                    width = self._x[c + 1] - self._x[c]
                    area = width * height
                    ratio = height / width
                    if ratio < 1.0:
                        ratio = 1 / ratio
                    reg: GroundRegion = GroundRegion(r, r, c, c, area, ratio)
                    more_regions = self._expand_rectangle(reg)
                    all_regions = all_regions | more_regions

        if len(all_regions) == 0:
            return None

        max_area = -1
        best_reg: Optional[GroundRegion] = None
        for reg in all_regions:
            if reg.area > max_area:
                max_area = reg.area  # type: ignore
                best_reg = reg

        # Occupy the cells
        for row in range(best_reg.rmin, best_reg.rmax + 1):  # type: ignore
            for col in range(best_reg.cmin, best_reg.cmax + 1):
                self._cells[row][col] = True

        x_center = (self._x[best_reg.cmin] + self._x[best_reg.cmax + 1]) / 2
        y_center = (self._y[best_reg.rmin] + self._y[best_reg.rmax + 1]) / 2
        width = self._x[best_reg.cmax + 1] - self._x[best_reg.cmin]
        height = self._y[best_reg.rmax + 1] - self._y[best_reg.rmin]
        kwargs = {KW_CENTER: Point(x_center, y_center), KW_SHAPE: Shape(width, height), KW_REGION: GROUND_REGION}
        return Rectangle(**kwargs)

    def _expand_rectangle(self, r: GroundRegion) -> Set[GroundRegion]:
        """
        Expands a rectangle of regions and generates all the valid regions.
        The expansion is done by increasing rows and columns.
        :param r: a ground region
        :return: the set of rectangles of ground regions
        """
        g_regions: Set[GroundRegion] = {r}
        pending: Deque[GroundRegion] = deque()
        pending.append(r)
        while len(pending) > 0:
            r = pending.popleft()
            if r.rmax < len(self._cells) - 1:  # Add one row
                row = r.rmax + 1
                valid = not any(self._cells[row][j] for j in range(r.cmin, r.cmax + 1))
                if valid:
                    height = self._y[r.rmax + 2] - self._y[r.rmin]
                    width = self._x[r.cmax + 1] - self._x[r.cmin]
                    area = height * width
                    ratio = height / width
                    if ratio < 1.0:
                        ratio = 1 / ratio
                    new_r = GroundRegion(r.rmin, r.rmax + 1, r.cmin, r.cmax, area, ratio)
                    if new_r not in g_regions:
                        g_regions.add(new_r)
                        pending.append(new_r)

            if r.cmax < len(self._cells[0]) - 1:  # Add one column
                col = r.cmax + 1
                valid = not any(self._cells[i][col] for i in range(r.rmin, r.rmax + 1))
                if valid:
                    height = self._y[r.rmax + 1] - self._y[r.rmin]
                    width = self._x[r.cmax + 2] - self._x[r.cmin]
                    area = height * width
                    ratio = height / width
                    if ratio < 1.0:
                        ratio = 1 / ratio
                    new_r = GroundRegion(r.rmin, r.rmax, r.cmin, r.cmax + 1, area, ratio)
                    if new_r not in g_regions:
                        g_regions.add(new_r)
                        pending.append(new_r)

        return g_regions


if __name__ == "__main__":
    L = Layout("../examples/basic.lay")
    print(L._ground_regions())
