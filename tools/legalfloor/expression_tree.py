from __future__ import annotations
from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Value
from math import sqrt as math_sqrt
from typing import Any, Callable
from enum import IntEnum

epsilon: ExpressionTree
debug_print: int = 0xFF


def turn_off_flag(flag: int):
    global debug_print
    debug_print &= ~flag


def turn_on_flag(flag: int):
    global debug_print
    debug_print |= flag


def debug(*values, flag: int = 0xFF):
    if flag & debug_print > 0:
        print(values)


def set_epsilon(new_epsilon: ExpressionTree):
    global epsilon
    epsilon = new_epsilon


def get_epsilon() -> float:
    return epsilon.evaluate()


class Cmp(IntEnum):
    LE = 0
    GE = 1
    EQ = 2


def add_equation(gekko: GEKKO, lhs: ExpressionTree, cmp: Cmp, rhs: ExpressionTree, name: str, hard: bool = False):
    lhs_expr = lhs.get_gekko_expression(lambda x: x, None, False)
    rhs_expr = rhs.get_gekko_expression(lambda x: x, None, False)
    e = epsilon.get_gekko_expression()
    lhs_val = lhs.evaluate()
    rhs_val = rhs.evaluate()
    e_val = epsilon.evaluate()
    if isinstance(lhs_expr, ExpressionTree) or isinstance(rhs_expr, ExpressionTree):
        raise Exception("?")
    if cmp is Cmp.LE:
        if hard:
            gekko.Equation(lhs_expr <= rhs_expr)
        else:
            gekko.Equation(lhs_expr <= rhs_expr + e)
        if lhs_val > rhs_val + e_val:
            debug("WARNING: Equation " + name + " is not met: ", lhs_val, "<=", rhs_val, flag=1)
        """
        else:
            print("Equation " + name + " is met: ", lhs_val, "<=", rhs_val, "(slack =",
                  rhs_val + epsilon - lhs_val, ")")
        """
    elif cmp is Cmp.GE:
        if hard:
            gekko.Equation(lhs_expr >= rhs_expr)
        else:
            gekko.Equation(lhs_expr >= rhs_expr - e)
        if lhs_val < rhs_val - e_val:
            debug("WARNING: Equation " + name + " is not met", lhs_val, ">=", rhs_val, flag=1)
        """
        else:
            print("Equation " + name + " is met: ", lhs_val, ">=", rhs_val, "(slack =",
                  lhs_val - rhs_val + epsilon, ")")
        """
    elif cmp is Cmp.EQ:
        if hard:
            gekko.Equation(lhs_expr == rhs_expr)
        else:
            gekko.Equation(lhs_expr >= rhs_expr - e)
            gekko.Equation(lhs_expr <= rhs_expr + e)
        if abs(lhs_val - rhs_val) > e_val:
            debug("WARNING: Equation " + name + " is not met", lhs_val, "==", rhs_val, flag=1)
        """
        else:
            print("Equation " + name + " is met: ", lhs_val, "==", rhs_val, "(slack = ",
                  epsilon - abs(lhs.evaluate() - rhs.evaluate()), ")")
        """


class NodeType(IntEnum):
    CST = 0
    VAR = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    EXP = 6
    SRT = 7


def value_of(v: GKVariable | float) -> float:
    if isinstance(v, float):
        return v
    if isinstance(v, list):
        return v[0]
    if isinstance(v, GKVariable):
        if type(v.value) == GK_Value:
            return v.value[0]
        if v.value is float:
            return v.value
    raise Exception("Unknown GekkoType ", type(v))


class ExpressionTree:
    value: Any
    gekko: GEKKO
    type: NodeType
    size: int

    def __init__(self, gekko: GEKKO,
                 value: int | float | GKVariable | list[ExpressionTree],
                 rtype: NodeType | None = None):
        self.gekko = gekko
        if rtype is None:
            if isinstance(value, float) or isinstance(value, int):
                self.value = float(value)
                self.type = NodeType.CST
                self.size = 1
            elif isinstance(value, GKVariable):
                self.value = value
                self.type = NodeType.VAR
                self.size = 1
            else:
                raise Exception("Unknown type for the base expression:", type(value))
        else:
            self.value = value
            self.type = rtype
            if isinstance(value, float) or isinstance(value, int) or isinstance(value, GKVariable):
                self.size = 1
            else:
                self.size = sum(map(lambda x: x.size, value)) + 1

    def assign(self, value: float):
        if self.type is not NodeType.VAR:
            raise Exception("You can only assign a value to a variable")
        self.value.value = [value]

    def fix_as_lower_bound(self):
        add_equation(self.gekko, self, Cmp.GE, ExpressionTree(self.gekko, self.evaluate()), "lower bound", hard=True)

    def get_string(self) -> str:
        symbols: list[str] = ["_", "_", "+", "-", "*", "/", "**", "_"]
        if self.type is NodeType.CST:
            return str(self.value)
        if self.type is NodeType.VAR:
            return self.value.name
        if self.type is NodeType.SRT:
            return "sqrt(" + self.value[0].get_string() + ")"
        return "(" + self.value[0].get_string() + ") " + symbols[self.type] + " (" + self.value[1].get_string() + ")"

    def get_gekko_expression_aux(self,
                                 getter: Callable[[Any], Any] = lambda x: x,
                                 root: Callable[[Any], Any] | None = None,
                                 aux_var: bool = True) -> tuple[GKVariable | float, ExpressionTree]:
        if root is None:
            root = self.gekko.sqrt
        blank: Callable[[Any, Any], Any] = lambda x, y: 0
        add: Callable[[Any, Any], Any] = lambda x, y: x + y
        sub: Callable[[Any, Any], Any] = lambda x, y: x - y
        mul: Callable[[Any, Any], Any] = lambda x, y: x * y
        div: Callable[[Any, Any], Any] = lambda x, y: x / y
        exp: Callable[[Any, Any], Any] = lambda x, y: x ** y
        operations: list[Callable[[Any, Any], Any]] = [blank, blank, add, sub, mul, div, exp, blank]
        if self.type is NodeType.CST or self.type is NodeType.VAR:
            return getter(self.value), ExpressionTree(self.gekko, self.value, self.type)
        if self.type is NodeType.SRT:
            expr, tree = self.value[0].get_gekko_expression_aux(getter, root, aux_var)
            if tree.size > 50 and aux_var:
                new_var = ExpressionTree(self.gekko, self.gekko.Var())
                new_var.value.value = [self.value[0].evaluate()]
                add_equation(self.gekko, new_var, Cmp.EQ, tree, "aux_var")
                return root(new_var), ExpressionTree(self.gekko, [new_var], self.type)
            return root(expr), ExpressionTree(self.gekko, [tree], self.type)
        lhs, tree1 = self.value[0].get_gekko_expression_aux(getter, root, aux_var)
        rhs, tree2 = self.value[1].get_gekko_expression_aux(getter, root, aux_var)
        if tree1.size + tree2.size > 50 and aux_var:
            lhs_var, rhs_var = ExpressionTree(self.gekko, self.gekko.Var()), \
                               ExpressionTree(self.gekko, self.gekko.Var())
            lhs_var.value.value = [self.value[0].evaluate()]
            rhs_var.value.value = [self.value[1].evaluate()]
            add_equation(self.gekko, lhs_var, Cmp.EQ, tree1, "aux_var")
            add_equation(self.gekko, rhs_var, Cmp.EQ, tree2, "aux_var")
            return operations[self.type](lhs_var.value, rhs_var.value), ExpressionTree(self.gekko, [lhs_var, rhs_var],
                                                                                       self.type)
        if isinstance(lhs, ExpressionTree):
            raise Exception("lhs is an expression tree!")
        if isinstance(rhs, ExpressionTree):
            raise Exception("rhs is an expression tree!")
        return operations[self.type](lhs, rhs), ExpressionTree(self.gekko, [tree1, tree2], self.type)

    def get_gekko_expression(self,
                             getter: Callable[[Any], Any] = lambda x: x,
                             root: Callable[[Any], Any] | None = None,
                             aux_var: bool = True) -> GKVariable | float:
        return self.get_gekko_expression_aux(getter, root, aux_var)[0]

    def evaluate(self) -> float:
        value = self.get_gekko_expression(value_of, math_sqrt, False)
        if not isinstance(value, float) and not isinstance(value, int):
            print(value)
            raise Exception("Invalid return of evaluate function!")
        return value

    def __add__(self, term: float | int | GKVariable | ExpressionTree) -> ExpressionTree:
        if isinstance(term, float) or isinstance(term, int) or isinstance(term, GKVariable):
            return self + ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("Term type is " + str(type(term)) + ", should be ExpressionTree!")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value + term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.ADD)

    def __sub__(self, term: float | int | GKVariable | ExpressionTree) -> ExpressionTree:
        if isinstance(term, float) or isinstance(term, int) or isinstance(term, GKVariable):
            return self - ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value - term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.SUB)

    def __mul__(self, term: float | int | GKVariable | ExpressionTree) -> ExpressionTree:
        if isinstance(term, float) or isinstance(term, int) or isinstance(term, GKVariable):
            return self * ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value * term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.MUL)

    def __truediv__(self, term: float | int | GKVariable | ExpressionTree) -> ExpressionTree:
        if isinstance(term, float) or isinstance(term, int) or isinstance(term, GKVariable):
            return self / ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value / term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.DIV)

    def __pow__(self, term: float | int | GKVariable | ExpressionTree) -> ExpressionTree:
        if isinstance(term, float) or isinstance(term, int) or isinstance(term, GKVariable):
            return self ** ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value ** term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.EXP)


def sqrt(et: ExpressionTree) -> ExpressionTree:
    if et.type is NodeType.CST:
        return ExpressionTree(et.gekko, math_sqrt(et.value))
    return ExpressionTree(et.gekko, [et], NodeType.SRT)
