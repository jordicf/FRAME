# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).
from random import randint
from time import time
from typing import Any, TypeVar, Union
from enum import IntEnum
from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Value

# import numpy as np
from frame.die.die import Die
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Rectangle
from tools.legalfloor.expression_tree import ExpressionTree, add_equation, Cmp, set_epsilon, get_epsilon, turn_off_flag
from tools.legalfloor.expression_tree import sqrt as expr_sqrt
from tools.rect.canvas import Canvas
from argparse import ArgumentParser

BoxType = tuple[float, float, float, float]
InputModule = tuple[BoxType, list[BoxType], list[BoxType], list[BoxType], list[BoxType]]
OptionalList = dict[int, float]
OptionalMatrix = dict[int, OptionalList]
HyperEdge = tuple[float, list[int]]
HyperGraph = list[HyperEdge]
GekkoType = Union[float, GKVariable]
FourVars = tuple[ExpressionTree, ExpressionTree, ExpressionTree, ExpressionTree]


# Think of GekkoType as either a constant, a variable or an expression.
# Aka. Something that evaluates to a number


def value_of(v: GekkoType) -> float:
    if v is float:
        return v
    if isinstance(v, list):
        return v[0]
    if isinstance(v, GKVariable):
        if type(v.value) == GK_Value:
            return v.value[0]
        if v.value is float:
            return v.value
    raise Exception("Unknown GekkoType")


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="A tool for module legalization", usage='%(prog)s [options]')
    parser.add_argument("netlist", type=str, help="Input netlist (.yaml)")
    parser.add_argument("die", type=str, help="Input die (.yaml)")
    parser.add_argument("--max_ratio", type=float, dest='max_ratio', default=2.00,
                        help="The maximum allowable ratio for a rectangle")
    parser.add_argument("--plot", dest="plot", const=True, default=False, action="store_const",
                        help="Plots the problem together with the solutions found")
    parser.add_argument("--verbose", dest="verbose", const=True, default=False, action="store_const",
                        help="Shows additional debug information")
    parser.add_argument("--ini_temp", dest="t0", type=float, default=0.9,
                        help="Initial annealing temperature")
    parser.add_argument("--alpha_temp", dest="dt", type=float, default=0.3,
                        help="Temperature annealing factor")
    parser.add_argument("--outfile", type=str, dest='file', default=None,
                        help="The output file path (yaml)")
    return vars(parser.parse_args(args))


def hsv_to_rgb(h: float, s: float = 0.7, v: float = 1.0) -> tuple[float, float, float]:
    h %= 1.0
    r = max(0.0, min(1.0, abs(6.0 * h - 3.0) - 1.0))
    g = max(0.0, min(1.0, 2.0 - abs(6.0 * h - 2.0)))
    b = max(0.0, min(1.0, 2.0 - abs(6.0 * h - 4.0)))

    r = r * s + 1 - s
    g = g * s + 1 - s
    b = b * s + 1 - s

    return r * v, g * v, b * v


def rgb_to_str(r: float, g: float, b: float) -> str:
    rs = hex(int(r * 255))[2:]
    gs = hex(int(g * 255))[2:]
    bs = hex(int(b * 255))[2:]
    if len(rs) < 2:
        rs = "0" + rs
    if len(gs) < 2:
        gs = "0" + gs
    if len(bs) < 2:
        bs = "0" + bs
    return "#" + rs + gs + bs


def hsv_to_str(h: float, s: float = 0.7, v: float = 1.0) -> str:
    r, g, b = hsv_to_rgb(h, s, v)
    return rgb_to_str(r, g, b)


T = TypeVar('T')


def optional_get(opt: dict[int, T] | None, index: int) -> T | None:
    if opt is None:
        return None
    if index in opt:
        return opt[index]
    return None


def smax(x: ExpressionTree, y: ExpressionTree, tau: ExpressionTree) -> ExpressionTree:
    half = ExpressionTree(x.gekko, 0.5)
    two = ExpressionTree(x.gekko, 2)
    four = ExpressionTree(x.gekko, 4)
    return half * (x + y + expr_sqrt((x - y) ** two + four * tau * tau))


U = TypeVar('U', float, ExpressionTree)


def thin(w: U, h: U) -> U:
    return w * h / (w * w + h * h)


class Cardinal(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class ModelModule:
    def __init__(self,
                 x: list[ExpressionTree],
                 y: list[ExpressionTree],
                 w: list[ExpressionTree],
                 h: list[ExpressionTree],
                 gekko: GEKKO,
                 trunk: BoxType,
                 die_width: float,
                 die_height: float,
                 max_ratio: float,
                 n_mods: int):
        self.N: list[int] = []
        self.S: list[int] = []
        self.E: list[int] = []
        self.W: list[int] = []
        self.x: list[ExpressionTree] = x
        self.y: list[ExpressionTree] = y
        self.w: list[ExpressionTree] = w
        self.h: list[ExpressionTree] = h
        self.c: int = 0
        self.id: int = n_mods
        self.dw: float = die_width
        self.dh: float = die_height
        self.area: ExpressionTree = ExpressionTree(gekko, 0)
        self.x_sum: ExpressionTree = ExpressionTree(gekko, 0)
        self.y_sum: ExpressionTree = ExpressionTree(gekko, 0)
        self.max_ratio: float = max_ratio
        self.degree = 0  # 0 = soft, 1 = hard, 2 = fixed. Just required for output purposes
        self.set_trunk(gekko, trunk)

    def set_degree(self, degree: int):
        if degree > self.degree:
            self.degree = degree

    def set_trunk(self, gekko: GEKKO, trunk: BoxType) -> None:
        assert self.c == 0, "!"
        self.c = 1
        self._define_vars(gekko, trunk)

    def add_rect_blank(self, gekko: GEKKO, rect: BoxType) -> FourVars:
        assert self.c > 0, "!"
        self.c += 1
        xv, yv, wv, hv = self._define_vars(gekko, rect)

        return xv, yv, wv, hv

    def add_rect_north(self, gekko: GEKKO, rect: BoxType) -> FourVars:
        xv, yv, wv, hv = self.add_rect_blank(gekko, rect)
        self.N.append(self.c - 1)

        # Keep the box attached
        hlf = ExpressionTree(gekko, 0.5)

        add_equation(gekko, yv, Cmp.EQ, self.y[0] + hlf * self.h[0] + hlf * hv, "north_attach")
        add_equation(gekko, xv, Cmp.GE, self.x[0] - hlf * self.w[0] + hlf * wv, "north_border0")
        add_equation(gekko, xv, Cmp.LE, self.x[0] + hlf * self.w[0] - hlf * wv, "north_border1")

        return xv, yv, wv, hv

    def add_rect_south(self, gekko: GEKKO, rect: BoxType) -> FourVars:
        xv, yv, wv, hv = self.add_rect_blank(gekko, rect)
        self.S.append(self.c - 1)

        # Keep the box attached
        hlf = ExpressionTree(gekko, 0.5)

        add_equation(gekko, yv, Cmp.EQ, self.y[0] - hlf * self.h[0] - hlf * hv, "south_attach")
        add_equation(gekko, xv, Cmp.GE, self.x[0] - hlf * self.w[0] + hlf * wv, "south_border0")
        add_equation(gekko, xv, Cmp.LE, self.x[0] + hlf * self.w[0] - hlf * wv, "south_border1")

        return xv, yv, wv, hv

    def add_rect_east(self, gekko: GEKKO, rect: BoxType) -> FourVars:
        xv, yv, wv, hv = self.add_rect_blank(gekko, rect)
        self.E.append(self.c - 1)

        # Keep the box attached
        hlf = ExpressionTree(gekko, 0.5)

        add_equation(gekko, xv, Cmp.EQ, self.x[0] + hlf * self.w[0] + hlf * wv, "east_attach")
        add_equation(gekko, yv, Cmp.GE, self.y[0] - hlf * self.h[0] + hlf * hv, "east_border0")
        add_equation(gekko, yv, Cmp.LE, self.y[0] + hlf * self.h[0] - hlf * hv, "east_border1")

        return xv, yv, wv, hv

    def add_rect_west(self, gekko: GEKKO, rect: BoxType) -> FourVars:
        xv, yv, wv, hv = self.add_rect_blank(gekko, rect)
        self.W.append(self.c - 1)

        # Keep the box attached
        hlf = ExpressionTree(gekko, 0.5)

        add_equation(gekko, xv, Cmp.EQ, self.x[0] - hlf * self.w[0] - hlf * wv, "west_attach")
        add_equation(gekko, yv, Cmp.GE, self.y[0] - hlf * self.h[0] + hlf * hv, "west_border0")
        add_equation(gekko, yv, Cmp.LE, self.y[0] + hlf * self.h[0] - hlf * hv, "west_border1")

        return xv, yv, wv, hv

    def _define_vars(self, gekko: GEKKO, rect: BoxType) -> FourVars:
        rect_id = len(self.x)
        var_x = ExpressionTree(gekko, gekko.Var(lb=0, ub=self.dw, name="x" + str(self.id) + "i" + str(rect_id)))
        var_y = ExpressionTree(gekko, gekko.Var(lb=0, ub=self.dh, name="y" + str(self.id) + "i" + str(rect_id)))
        var_w = ExpressionTree(gekko, gekko.Var(lb=0.1, ub=self.dw, name="w" + str(self.id) + "i" + str(rect_id)))
        var_h = ExpressionTree(gekko, gekko.Var(lb=0.1, ub=self.dh, name="h" + str(self.id) + "i" + str(rect_id)))
        var_x.value.value = [rect[0]]
        var_y.value.value = [rect[1]]
        var_w.value.value = [rect[2]]
        var_h.value.value = [rect[3]]
        self.x.append(var_x)
        self.y.append(var_y)
        self.w.append(var_w)
        self.h.append(var_h)

        self.area += var_w * var_h
        self.x_sum += var_x * var_w * var_h
        self.y_sum += var_y * var_w * var_h

        # The box must stay inside the dice
        hlf = ExpressionTree(gekko, 0.5)

        add_equation(gekko, var_x - hlf * var_w, Cmp.GE, ExpressionTree(gekko, 0), "bounds_left")
        add_equation(gekko, var_y - hlf * var_h, Cmp.GE, ExpressionTree(gekko, 0), "bounds_bottom")
        add_equation(gekko, var_x + hlf * var_w, Cmp.LE, ExpressionTree(gekko, self.dw), "bounds_right")
        add_equation(gekko, var_y + hlf * var_h, Cmp.LE, ExpressionTree(gekko, self.dh), "bounds_top")

        # The ratio cannot exceed a maximum value
        add_equation(gekko, thin(var_w, var_h), Cmp.GE, ExpressionTree(gekko, thin(self.max_ratio, 1)), "ratio")

        return var_x, var_y, var_w, var_h


class Model:
    """GEKKO model with variables"""
    gekko: GEKKO

    M: list[ModelModule]
    dw: float  # Die width
    dh: float  # Die height

    # Model variables
    # (without accurate type hints because GEKKO does not have type hints yet)
    x: list[list[ExpressionTree]]
    y: list[list[ExpressionTree]]
    w: list[list[ExpressionTree]]
    h: list[list[ExpressionTree]]
    time: ExpressionTree
    alpha: ExpressionTree
    max_ratio: float

    og_names: list[str]
    og_area: list[float]
    hyper: HyperGraph
    enforces: list[tuple[ExpressionTree, float]]

    fixed_t: float
    tau: ExpressionTree

    # For visualization
    hue_array: list[str]
    output_counter: int

    temperature_decay: float
    temperature_ini: float

    def time_advance(self, amount: float):
        if amount <= 0:
            raise Exception("The amount of time must be > 0")
        self.time.assign(self.time.evaluate() + amount)
        self.time.fix_as_lower_bound()

    def define_time(self) -> None:
        t = self.gekko.Var([0])
        self.time = ExpressionTree(self.gekko, t)
        set_epsilon((ExpressionTree(self.gekko, self.temperature_decay) ** self.time) * self.temperature_ini)
        self.time_advance(self.fixed_t)
        print("")

    def define_module(self, trunk_box: BoxType) -> int:
        x: list[ExpressionTree] = []
        y: list[ExpressionTree] = []
        w: list[ExpressionTree] = []
        h: list[ExpressionTree] = []
        m = ModelModule(x, y, w, h, self.gekko, trunk_box, self.dw, self.dh, self.max_ratio, len(self.M))
        self.M.append(m)
        self.x.append(x)
        self.y.append(y)
        self.w.append(w)
        self.h.append(h)
        return len(self.M) - 1

    def add_rect(self, m: int, box: BoxType, direction: Cardinal) -> int:
        call = self.M[m].add_rect_blank
        if direction is Cardinal.NORTH:
            call = self.M[m].add_rect_north
        elif direction is Cardinal.SOUTH:
            call = self.M[m].add_rect_south
        elif direction is Cardinal.EAST:
            call = self.M[m].add_rect_east
        elif direction is Cardinal.WEST:
            call = self.M[m].add_rect_west
        # xv, yv, wv, hv =
        call(self.gekko, box)
        return len(self.M) - 1

    def fix(self,
            m: int,
            xl: OptionalList | None,
            yl: OptionalList | None,
            wl: OptionalList | None,
            hl: OptionalList | None) -> None:
        for i in range(0, self.M[m].c):
            x_get: float | None = optional_get(xl, i)
            y_get: float | None = optional_get(yl, i)
            w_get: float | None = optional_get(wl, i)
            h_get: float | None = optional_get(hl, i)
            x_const: ExpressionTree = ExpressionTree(self.gekko, 0)
            y_const: ExpressionTree = ExpressionTree(self.gekko, 0)
            w_const: ExpressionTree = ExpressionTree(self.gekko, 0)
            h_const: ExpressionTree = ExpressionTree(self.gekko, 0)
            if x_get is not None:
                x_const += x_get
            if y_get is not None:
                y_const += y_get
            if w_get is not None:
                w_const += w_get
            if h_get is not None:
                h_const += h_get
            if i != 0 and isinstance(x_get, float):
                x_const += self.M[m].x[0]
                y_const += self.M[m].y[0]
            elif x_get is not None:
                self.M[m].set_degree(2)
            if x_get is not None:
                self.enforces.append((self.M[m].x[i], x_const.evaluate()))
                add_equation(self.gekko, self.M[m].x[i], Cmp.EQ, x_const, "fix_x")
            if y_get is not None:
                self.enforces.append((self.M[m].y[i], y_const.evaluate()))
                add_equation(self.gekko, self.M[m].y[i], Cmp.EQ, y_const, "fix_y")
            if w_get is not None:
                self.enforces.append((self.M[m].w[i], w_const.evaluate()))
                self.M[m].set_degree(1)
                add_equation(self.gekko, self.M[m].w[i], Cmp.EQ, w_const, "fix_w")
            if h_get is not None:
                self.enforces.append((self.M[m].h[i], h_const.evaluate()))
                add_equation(self.gekko, self.M[m].h[i], Cmp.EQ, h_const, "fix_h")

    def objective(self) -> ExpressionTree:
        maximum_size = float('inf')
        obj = ExpressionTree(self.gekko, 0.0)
        for (weight, Set) in self.hyper:
            centroid_x: ExpressionTree = ExpressionTree(self.gekko, 0.0)
            centroid_y: ExpressionTree = ExpressionTree(self.gekko, 0.0)
            for i in Set:
                centroid_x += self.M[i].x_sum / self.M[i].area
                centroid_y += self.M[i].y_sum / self.M[i].area
                for j in range(0, self.M[i].c):
                    obj += (
                            (self.M[i].x[j] - self.M[i].x_sum / self.M[i].area) ** 2 +
                            (self.M[i].y[j] - self.M[i].y_sum / self.M[i].area) ** 2
                    ) * (weight * weight)
                    if obj.size >= maximum_size:
                        return obj
            centroid_x /= len(Set)
            centroid_y /= len(Set)
            for i in Set:
                obj += (
                    (self.M[i].x_sum / self.M[i].area - centroid_x) ** 2 +
                    (self.M[i].y_sum / self.M[i].area - centroid_y) ** 2
                ) * (weight * weight)
                if obj.size >= maximum_size:
                    return obj
        return obj

    def reinforce_fixed(self):
        for (x, v) in self.enforces:
            x.value.value = [v]

    def build_model(self):
        ml = self.ml
        al = self.al
        xl = self.xl
        yl = self.yl
        wl = self.wl
        hl = self.hl
        die_width = self.die_width
        die_height = self.die_height
        hyper = self.hyper
        max_ratio = self.max_ratio
        og_names = self.og_names

        self.M: list[ModelModule] = []
        self.x: list[list[ExpressionTree]] = []
        self.y: list[list[ExpressionTree]] = []
        self.w: list[list[ExpressionTree]] = []
        self.h: list[list[ExpressionTree]] = []
        self.dw: float = die_width
        self.dh: float = die_height
        self.max_ratio: float = max_ratio
        self.og_names: list[str] = og_names
        self.og_area: list[float] = al
        self.hyper = hyper

        self.enforces: list[tuple[ExpressionTree, float]] = []

        """Constructs the GEKKO object and initializes the model"""
        self.gekko = GEKKO(remote=False)
        # self.gekko.options.SOLVER = 3
        # self.gekko.options.IMODE = 6
        self.gekko.options.COLDSTART = 1
        self.gekko.options.MAX_ITER = 1

        # self.tau = self.gekko.Var(lb = 0, name="tau")
        # self.tau.value = [5]
        self.tau = ExpressionTree(self.gekko, 0.01 * min(die_width, die_height) / len(ml))
        self.define_time()

        # Variable definition
        for (trunk, Nb, Sb, Eb, Wb) in ml:
            m = self.define_module(trunk)
            for box_i in Nb:
                self.add_rect(m, box_i, Cardinal.NORTH)
            for box_i in Sb:
                self.add_rect(m, box_i, Cardinal.SOUTH)
            for box_i in Eb:
                self.add_rect(m, box_i, Cardinal.EAST)
            for box_i in Wb:
                self.add_rect(m, box_i, Cardinal.WEST)

        # Minimal area requirements
        for m in range(0, len(al)):
            add_equation(self.gekko, self.M[m].area, Cmp.GE, ExpressionTree(self.gekko, al[m]), "min_area")

        # No Intra-Module Intersection
        hlf = ExpressionTree(self.gekko, 0.5)
        two = ExpressionTree(self.gekko, 2)
        qrt = ExpressionTree(self.gekko, 0.25)
        zero = ExpressionTree(self.gekko, 0)

        for m in range(0, len(al)):
            nid = self.M[m].N
            sid = self.M[m].S
            eid = self.M[m].E
            wid = self.M[m].W

            nid.sort(key=lambda z: self.M[m].x[z].evaluate())
            for i in range(0, len(nid) - 1):
                x, y = nid[i], nid[i + 1]
                add_equation(self.gekko, self.M[m].x[x] + hlf * self.M[m].w[x], Cmp.LE,
                             self.M[m].x[y] - hlf * self.M[m].w[y], "no_intra" + "module_north_intersection")

            sid.sort(key=lambda z: self.M[m].x[z].evaluate())
            for i in range(0, len(sid) - 1):
                x, y = sid[i], sid[i + 1]
                add_equation(self.gekko, self.M[m].x[x] + hlf * self.M[m].w[x], Cmp.LE,
                             self.M[m].x[y] - hlf * self.M[m].w[y], "no_intra" + "module_south_intersection")

            eid.sort(key=lambda z: self.M[m].y[z].evaluate())
            for i in range(0, len(eid) - 1):
                x, y = eid[i], eid[i + 1]
                add_equation(self.gekko, self.M[m].y[x] + hlf * self.M[m].h[x], Cmp.LE,
                             self.M[m].y[y] - hlf * self.M[m].h[y], "no_intra" + "module_east_intersection")

            wid.sort(key=lambda z: self.M[m].y[z].evaluate())
            for i in range(0, len(wid) - 1):
                x, y = wid[i], wid[i + 1]
                add_equation(self.gekko, self.M[m].y[x] + hlf * self.M[m].h[x], Cmp.LE,
                             self.M[m].y[y] - hlf * self.M[m].h[y], "no_intra" + "module_west_intersection")

        # No Inter-Module Intersection
        for m in range(0, len(al)):
            for n in range(m + 1, len(al)):
                for i in range(0, self.M[m].c):
                    for j in range(0, self.M[n].c):
                        t1 = (self.x[m][i] - self.x[n][j]) ** two - qrt * (self.w[m][i] + self.w[n][j]) ** two
                        t2 = (self.y[m][i] - self.y[n][j]) ** two - qrt * (self.h[m][i] + self.h[n][j]) ** two
                        add_equation(self.gekko, smax(t1, t2, self.tau), Cmp.GE, zero, "no_inter" + "module_overlap")

        # Fixed/Hard modules
        for m in range(0, len(al)):
            self.fix(m, optional_get(xl, m), optional_get(yl, m), optional_get(wl, m), optional_get(hl, m))

        # Objective function
        # print("Size of objective: ", self.objective().size)
        self.apply_objective_function()

    def set_ml(self, ml: list[InputModule]):
        self.ml = ml

    def set_fixed_t(self, ft: float):
        self.fixed_t = ft

    def __init__(self,
                 ml: list[InputModule],
                 al: list[float],
                 xl: OptionalMatrix,
                 yl: OptionalMatrix,
                 wl: OptionalMatrix,
                 hl: OptionalMatrix,
                 die_width: float,
                 die_height: float,
                 hyper: HyperGraph,
                 max_ratio: float,
                 og_names: list[str],
                 temp0: float,
                 alpha_temp: float):
        assert len(ml) == len(al), "M and A need to have the same length!"
        self.temperature_decay = temp0 #0.9
        self.temperature_ini = alpha_temp #0.3
        self.ml = ml
        self.al = al
        self.xl = xl
        self.yl = yl
        self.wl = wl
        self.hl = hl
        self.die_width = die_width
        self.die_height = die_height
        self.hyper = hyper
        self.max_ratio = max_ratio
        self.og_names = og_names
        self.hue_array = []
        self.output_counter = 0
        self.fixed_t = 1
        self.build_model()

    def apply_objective_function(self):
        self.gekko.Obj((self.objective() + self.tau).get_gekko_expression())

    def interactive_draw(self, canvas_width=500, canvas_height=500) -> None:
        canvas = Canvas(width=canvas_width, height=canvas_height)
        canvas.clear(col="#000000")
        canvas.set_coords(-1, -1, self.dw + 1, self.dh + 1)

        if len(self.hue_array) != len(self.M):
            self.hue_array = []
            for i in range(0, len(self.M)):
                hue = i / len(self.M)
                self.hue_array.append(hsv_to_str(hue) + "90")
            for i in range(0, len(self.M)):
                j = randint(i, len(self.M) - 1)
                self.hue_array[i], self.hue_array[j] = self.hue_array[j], self.hue_array[i]

        for i in range(0, len(self.M)):
            m = self.M[i]
            for j in range(0, len(m.x)):
                a = m.x[j].evaluate() - 0.5 * m.w[j].evaluate()
                b = m.y[j].evaluate() - 0.5 * m.h[j].evaluate()
                c = m.x[j].evaluate() + 0.5 * m.w[j].evaluate()
                d = m.y[j].evaluate() + 0.5 * m.h[j].evaluate()
                canvas.drawbox(((a, b), (c, d)), col=self.hue_array[i])

        canvas.drawbox(((0, 0), (self.dw, self.dh)), "#00000000", "#FFFFFF")

        for hyper_edge in self.hyper:
            modules = hyper_edge[1]
            x_center = 0.0
            y_center = 0.0
            list_x = []
            list_y = []
            for module in modules:
                x_sum = 0.0
                y_sum = 0.0
                area = 0.0
                for rect_id in range(0, self.M[module].c):
                    r_area = self.M[module].w[rect_id].evaluate() * self.M[module].h[rect_id].evaluate()
                    x_sum += self.M[module].x[rect_id].evaluate() * r_area
                    y_sum += self.M[module].y[rect_id].evaluate() * r_area
                    area += r_area
                x, y = x_sum / area, y_sum / area
                for rect_id in range(0, self.M[module].c):
                    x_r = self.M[module].x[rect_id].evaluate()
                    y_r = self.M[module].y[rect_id].evaluate()
                    canvas.dot((x_r, y_r), color="#FFFFFF", dot_type="thin_circle")
                    canvas.line(((x_r, y_r), (x, y)), color="#FFFFFF", thickness=1, line_type="dashed")
                canvas.dot((x, y), color="#FFFFFF", dot_type="solid_circle")
                x_center += x / len(modules)
                y_center += y / len(modules)
                list_x.append(x)
                list_y.append(y)
            for i in range(0, len(list_x)):
                canvas.line(((x_center, y_center), (list_x[i], list_y[i])), color="#FFFFFF")

        canvas.save("./example_visuals/frame" + str(self.output_counter) + ".png")
        self.output_counter += 1

    def solve(self, verbose=False) -> None:
        self.gekko.options.COLDSTART = 1
        self.gekko.options.MAX_ITER += 1000
        self.gekko.solve(disp=verbose, debug=0)
        self.reinforce_fixed()

    def get_netlist(self) -> Netlist:
        yaml = "Modules: {\n"
        for i in range(0, len(self.M)):
            if i != 0:
                yaml += ",\n"
            yaml += "  " + self.og_names[i] + ": {\n"
            if self.M[i].degree == 0:
                yaml += "    area: " + str(self.og_area[i]) + ",\n"
            elif self.M[i].degree == 1:
                yaml += "    hard: true,\n"
            else:
                yaml += "    fixed: true,\n"
            yaml += "    rectangles: ["
            for j in range(0, len(self.M[i].x)):
                rect = [self.M[i].x[j].evaluate(),
                        self.M[i].y[j].evaluate(),
                        self.M[i].w[j].evaluate(),
                        self.M[i].h[j].evaluate()]
                if j != 0:
                    yaml += ", "
                yaml += str(rect)
            yaml += "]\n  }"
        yaml += "\n}\nNets: [\n  "
        for i in range(0, len(self.hyper)):
            if i != 0:
                yaml += ",\n  ["
            else:
                yaml += "["
            for j in range(0, len(self.hyper[i][1])):
                if j != 0:
                    yaml += ", "
                m = self.hyper[i][1][j]
                if m < len(self.og_names):
                    yaml += self.og_names[m]
                else:
                    yaml += "__fixed_region_" + str(m - len(self.og_names))
            yaml += "]"
        yaml += "\n]\n"
        # print(yaml)
        return Netlist(yaml)

    def is_solved(self):
        return self.gekko.options.APPSTATUS == 1


def netlist_to_utils(netlist: Netlist):
    ml: list[InputModule] = []
    al: list[float] = []
    xl: OptionalMatrix = {}
    yl: OptionalMatrix = {}
    wl: OptionalMatrix = {}
    hl: OptionalMatrix = {}
    mod_map: dict[str, int] = {}
    og_names = []
    for module in netlist.modules:
        mod_map[module.name] = len(ml)
        og_names.append(module.name)
        b: InputModule = ((0, 0, 0, 0), [], [], [], [])
        trunk_defined = False
        for rect in module.rectangles:
            r = (rect.center.x, rect.center.y, rect.shape.w, rect.shape.h)
            if rect.location == Rectangle.StogLocation.TRUNK:
                b = (r, b[1], b[2], b[3], b[4])
                trunk_defined = True
            elif rect.location == Rectangle.StogLocation.NORTH:
                b[1].append(r)
            elif rect.location == Rectangle.StogLocation.SOUTH:
                b[2].append(r)
            elif rect.location == Rectangle.StogLocation.EAST:
                b[3].append(r)
            elif rect.location == Rectangle.StogLocation.WEST:
                b[4].append(r)
            elif not trunk_defined:
                b = (r, b[1], b[2], b[3], b[4])
                trunk_defined = True
            else:
                b[1].append(r)
        if module.is_hard:
            xl[len(ml)] = {}
            yl[len(ml)] = {}
            wl[len(ml)] = {}
            hl[len(ml)] = {}
        if module.is_fixed:
            xl[len(ml)][0] = b[0][0]
            yl[len(ml)][0] = b[0][1]
        if module.is_hard:
            wl[len(ml)][0] = b[0][2]
            hl[len(ml)][0] = b[0][3]
            i = 1
            for q in range(1, 5):
                bq = b[q]
                if isinstance(bq, list):
                    for (x, y, w, h) in bq:
                        xl[len(ml)][i] = x
                        yl[len(ml)][i] = y
                        wl[len(ml)][i] = w
                        hl[len(ml)][i] = h
                        i += 1
        ml.append(b)
        al.append(module.area())

    hyper: HyperGraph = []
    for edge in netlist.edges:
        connection = list(map(lambda e_mod: mod_map[e_mod.name], edge.modules))
        weight = edge.weight
        hyper.append((weight, connection))
    return ml, al, xl, yl, wl, hl, hyper, og_names


def compute_options(options) -> tuple[list[InputModule],  # Module list
                                      list[float],  # Area list
                                      OptionalMatrix,  # X coords
                                      OptionalMatrix,  # Y coords
                                      OptionalMatrix,  # widths
                                      OptionalMatrix,  # heights
                                      float,  # Die width
                                      float,  # Die height
                                      HyperGraph,  # Hypergraph
                                      float,  # Max ratio
                                      list[str]]:  # Original manes
    max_ratio: float = options['max_ratio']

    die = Die(options['die'])
    die_width: float = die.width
    die_height: float = die.height
    netlist = Netlist(options['netlist'])
    ml, al, xl, yl, wl, hl, hyper, og_names = netlist_to_utils(netlist)
    return ml, al, xl, yl, wl, hl, die_width, die_height, hyper, max_ratio, og_names


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """
    Main function.
    """
    options = parse_options(prog, args)
    ml, al, xl, yl, wl, hl, die_width, die_height, hyper, max_ratio, og_names = compute_options(options)
    m = Model(ml, al, xl, yl, wl, hl, die_width, die_height, hyper, max_ratio, og_names, options['t0'], options['dt'])
    turn_off_flag(1)

    print("Initial cost: ", m.objective().evaluate())

    if options['verbose']:
        m.gekko.open_folder()
    if options['plot']:
        m.interactive_draw()

    start = time()

    # noinspection PyBroadException
    #try:
    if True:
        m.apply_objective_function()
        for i in range(0, 100, 1):  # type: ignore
            t = m.time.evaluate()
            print("time t =", t)
            print("epsilon =", get_epsilon())

            m.set_fixed_t(i + 1)
            m.build_model()
            m.solve(options['verbose'])
            ntuple = netlist_to_utils(m.get_netlist())
            ml = ntuple[0]
            m.set_ml(ml)

            #m.time_advance(1)
            if options['plot']:
                m.interactive_draw()
            if m.is_solved() and get_epsilon() < 1e-10:
                print("")
                break
    #except Exception:
    #    print("No solution was found!")

    end = time()
    print("Final cost: ", m.objective().evaluate())
    print("Elapsed time: ", end - start, "s")

    net = m.get_netlist()
    if options['file'] is None:
        print(net)
    else:
        f = open(options['file'], "w")
        f.write(net)
        f.close()
    return 0


if __name__ == "__main__":
    main()
