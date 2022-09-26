from __future__ import annotations
from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Value
from math import sqrt as math_sqrt
from typing import Any, Callable
from enum import IntEnum


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
                                 aux_var: bool = True) -> tuple[GKVariable | float, int]:
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
            return getter(self.value), 1
        if self.type is NodeType.SRT:
            expr, size = self.value[0].get_gekko_expression_aux(getter, root, aux_var)
            if size > 50 and aux_var:
                new_var = self.gekko.Var()
                self.gekko.Equation(new_var == expr)
                return root(new_var), 2
            return root(expr), size + 1
        lhs, size1 = self.value[0].get_gekko_expression_aux(getter, root, aux_var)
        rhs, size2 = self.value[1].get_gekko_expression_aux(getter, root, aux_var)
        if size1 + size2 > 50 and aux_var:
            lhs_var, rhs_var = self.gekko.Var(), self.gekko.Var()
            self.gekko.Equation(lhs_var == lhs)
            self.gekko.Equation(rhs_var == rhs)
            return operations[self.type](lhs_var, rhs_var), 3
        return operations[self.type](lhs, rhs), size1 + size2 + 1

    def get_gekko_expression(self,
                             getter: Callable[[Any], Any] = lambda x: x,
                             root: Callable[[Any], Any] | None = None,
                             auxvar: bool = True) -> GKVariable | float:
        return self.get_gekko_expression_aux(getter, root, auxvar)[0]

    def evaluate(self) -> float:
        return self.get_gekko_expression(value_of, math_sqrt, False)

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
