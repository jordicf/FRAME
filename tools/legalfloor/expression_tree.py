from __future__ import annotations
from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Value
from gekko.gk_parameter import GK_MV
from math import sqrt as math_sqrt
from typing import Any, Callable, Optional
from enum import IntEnum

epsilon: Optional['ExpressionTree'] = None
debug_print: int = 0xFF
named_variables: set[str] = set()


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
    epsilon.is_epsilon = True


def get_epsilon() -> float:
    assert epsilon is not None
    return epsilon.evaluate()


def set_epsilon_gekko(gekko: GEKKO):
    assert epsilon is not None
    epsilon.set_gekko(gekko)


class Cmp(IntEnum):
    LE = 0
    GE = 1
    EQ = 2


def add_equation(
    gekko: GEKKO,
    lhs: ExpressionTree,
    cmp: Cmp,
    rhs: ExpressionTree,
    name: str,
    hard: bool = False,
) -> None:
    lhs_expr = lhs.get_gekko_expression(lambda x: x, None, False)
    rhs_expr = rhs.get_gekko_expression(lambda x: x, None, False)
    assert epsilon is not None
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
            debug(
                "WARNING: Equation " + name + " is not met: ",
                lhs_val,
                "<=",
                rhs_val,
                flag=1,
            )
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
            debug(
                "WARNING: Equation " + name + " is not met",
                lhs_val,
                ">=",
                rhs_val,
                flag=1,
            )
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
            debug(
                "WARNING: Equation " + name + " is not met",
                lhs_val,
                "==",
                rhs_val,
                flag=1,
            )
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


def value_of(v: GKVariable | GK_MV | float) -> float:
    if isinstance(v, float):
        return v
    if isinstance(v, list):
        return v[0]
    if isinstance(v, GKVariable) or isinstance(v, GK_MV):
        if isinstance(v.value, float) or isinstance(v.value, int):
            return float(v.value)
        if isinstance(v.value.value, float) or isinstance(v.value.value, int):
            return float(v.value.value)
        if hasattr(v.value.value, "__getitem__"):
            if isinstance(v.value.value, GK_Value):
                if isinstance(v.value.value.value, float) or isinstance(
                    v.value.value.value, int
                ):
                    return float(v.value.value.value)
            return v.value.value[0]
    raise Exception("Unknown GekkoType ", type(v))


class ExpressionTree:
    value: Any
    gekko: GEKKO
    type: NodeType
    size: int
    data: None | dict[str, Any]
    unary_equations: dict[str, Equation]
    previous_value: None | float
    is_epsilon: bool = False

    def __init__(
        self,
        gekko: GEKKO,
        value: int | float | GKVariable | GK_MV | list[ExpressionTree],
        rtype: NodeType | None = None,
        data: None | dict[str, Any] = None,
        is_epsilon: bool = False,
    ):
        self.data = data
        self.gekko = gekko
        self.unary_equations = dict()
        self.previous_value = None
        self.is_epsilon = is_epsilon
        if rtype is None:
            if isinstance(value, float) or isinstance(value, int):
                self.value = float(value)
                self.type = NodeType.CST
                self.size = 1
            elif isinstance(value, GKVariable) or isinstance(value, GK_MV):
                self.value = value
                self.type = NodeType.VAR
                self.size = 1
            else:
                raise Exception("Unknown type for the base expression:", type(value))
        else:
            self.value = value
            self.type = rtype
            if (
                isinstance(value, float)
                or isinstance(value, int)
                or isinstance(value, GKVariable)
                or isinstance(value, GK_MV)
            ):
                self.size = 1
            else:
                self.size = sum(map(lambda x: x.size, value)) + 1
        if isinstance(value, GKVariable) or isinstance(value, GK_MV):
            self.previous_value = value.value

    @staticmethod
    def create_variable(
        gekko: GEKKO, value: float, lb: float = 0, ub: float = 1, name=""
    ):
        real_name = name
        index = 0
        while real_name in named_variables:
            real_name = "%s_%i" % (name, index)
            index += 1
        mv = gekko.Var(value=value, lb=lb, ub=ub, name=name, integer=False)
        # mv.STATUS = 1
        return ExpressionTree(
            gekko, mv, data={"lb": lb, "ub": ub, "name": real_name, "integer": False}
        )

    def get_variable_list(
        self, repeats: set[str] | None = None
    ) -> list[ExpressionTree]:
        if repeats is None:
            repeats = set()
        if self.type == NodeType.VAR:
            assert self.data is not None
            if self.data["name"] not in repeats:
                repeats.add(self.data["name"])
                return [self]
            return []
        elif isinstance(self.value, list):
            res = []
            for val in self.value:
                res += val.get_variable_list(repeats)
            return res
        else:
            return []

    def assign(self, value: float):
        if self.type is not NodeType.VAR:
            raise Exception("You can only assign a value to a variable")
        self.value.value = [value]

    def undo(self):
        if isinstance(self.value, list):
            for x in self.value:
                x.undo()
        elif isinstance(self.value, GKVariable) or isinstance(self.value, GK_MV):
            self.value.value = self.previous_value

    def set_gekko(self, gekko: GEKKO):
        if self.gekko == gekko:
            return
        self.gekko = gekko
        if isinstance(self.value, list):
            for x in self.value:
                assert isinstance(x, ExpressionTree)
                x.set_gekko(gekko)
        elif isinstance(self.value, GKVariable) or isinstance(self.value, GK_MV):
            self.previous_value = self.value.value
            if self.data is None:
                new_value = gekko.Var(value=self.previous_value, integer=False)
            else:
                new_value = gekko.Var(value=self.previous_value, **self.data)
            # new_value.STATUS = 1
            # new_value.value = self.value.value
            self.value = new_value

    def get_string(self) -> str:
        symbols: list[str] = ["_", "_", "+", "-", "*", "/", "**", "_"]
        if self.type is NodeType.CST:
            return str(self.value)
        if self.type is NodeType.VAR:
            return self.value.name
        if self.type is NodeType.SRT:
            return "sqrt(" + self.value[0].get_string() + ")"
        return (
            "("
            + self.value[0].get_string()
            + ") "
            + symbols[self.type]
            + " ("
            + self.value[1].get_string()
            + ")"
        )

    def __repr__(self) -> str:
        return self.get_string()

    def get_gekko_expression_aux(
        self,
        getter: Callable[[Any], Any] = lambda x: x,
        root: Callable[[Any], Any] | None = None,
        aux_var: bool = True,
    ) -> tuple[GKVariable | GK_MV | float, ExpressionTree]:
        if root is None:
            root = self.gekko.sqrt
        blank: Callable[[Any, Any], Any] = lambda x, y: 0
        add: Callable[[Any, Any], Any] = lambda x, y: x + y
        sub: Callable[[Any, Any], Any] = lambda x, y: x - y
        mul: Callable[[Any, Any], Any] = lambda x, y: x * y
        div: Callable[[Any, Any], Any] = lambda x, y: x / y
        exp: Callable[[Any, Any], Any] = lambda x, y: x**y
        operations: list[Callable[[Any, Any], Any]] = [
            blank,
            blank,
            add,
            sub,
            mul,
            div,
            exp,
            blank,
        ]
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
            lhs_var, rhs_var = ExpressionTree(
                self.gekko, self.gekko.Var()
            ), ExpressionTree(self.gekko, self.gekko.Var())
            lhs_var.value.value = [self.value[0].evaluate()]
            rhs_var.value.value = [self.value[1].evaluate()]
            add_equation(self.gekko, lhs_var, Cmp.EQ, tree1, "aux_var")
            add_equation(self.gekko, rhs_var, Cmp.EQ, tree2, "aux_var")
            return operations[self.type](lhs_var.value, rhs_var.value), ExpressionTree(
                self.gekko, [lhs_var, rhs_var], self.type
            )
        if isinstance(lhs, ExpressionTree):
            raise Exception("lhs is an expression tree!")
        if isinstance(rhs, ExpressionTree):
            raise Exception("rhs is an expression tree!")
        return operations[self.type](lhs, rhs), ExpressionTree(
            self.gekko, [tree1, tree2], self.type
        )

    def get_gekko_expression(
        self,
        getter: Callable[[Any], Any] = lambda x: x,
        root: Callable[[Any], Any] | None = None,
        aux_var: bool = True,
    ) -> GKVariable | GK_MV | float:
        if self.is_epsilon:
            value, _ = self.get_gekko_expression_aux(value_of, math_sqrt, False)
            if not isinstance(value, float) and not isinstance(value, int):
                print(value)
                raise Exception("Invalid return of evaluate function!")
            if float(value) < 1e-6:
                return 0.0
            return float(value)
        return self.get_gekko_expression_aux(getter, root, aux_var)[0]

    def evaluate(self) -> float:
        value = self.get_gekko_expression(value_of, math_sqrt, False)
        if not isinstance(value, float) and not isinstance(value, int):
            print(value)
            raise Exception("Invalid return of evaluate function!")
        return value

    def __add__(
        self, term: float | int | GKVariable | GK_MV | ExpressionTree
    ) -> ExpressionTree:
        if (
            isinstance(term, float)
            or isinstance(term, int)
            or isinstance(term, GKVariable)
            or isinstance(term, GK_MV)
        ):
            return self + ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception(
                "Term type is " + str(type(term)) + ", should be ExpressionTree!"
            )
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value + term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.ADD)

    def __sub__(
        self, term: float | int | GKVariable | GK_MV | ExpressionTree
    ) -> ExpressionTree:
        if (
            isinstance(term, float)
            or isinstance(term, int)
            or isinstance(term, GKVariable)
            or isinstance(term, GK_MV)
        ):
            return self - ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value - term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.SUB)

    def __mul__(
        self, term: float | int | GKVariable | GK_MV | ExpressionTree
    ) -> ExpressionTree:
        if (
            isinstance(term, float)
            or isinstance(term, int)
            or isinstance(term, GKVariable)
            or isinstance(term, GK_MV)
        ):
            return self * ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value * term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.MUL)

    def __truediv__(
        self, term: float | int | GKVariable | GK_MV | ExpressionTree
    ) -> ExpressionTree:
        if (
            isinstance(term, float)
            or isinstance(term, int)
            or isinstance(term, GKVariable)
            or isinstance(term, GK_MV)
        ):
            return self / ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value / term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.DIV)

    def __pow__(
        self, term: float | int | GKVariable | GK_MV | ExpressionTree
    ) -> ExpressionTree:
        if (
            isinstance(term, float)
            or isinstance(term, int)
            or isinstance(term, GKVariable)
            or isinstance(term, GK_MV)
        ):
            return self ** ExpressionTree(self.gekko, term)
        elif not isinstance(term, ExpressionTree):
            raise Exception("")
        if self.type is NodeType.CST and term.type is NodeType.CST:
            return ExpressionTree(self.gekko, self.value**term.value, NodeType.CST)
        return ExpressionTree(self.gekko, [self, term], NodeType.EXP)


def sqrt(et: ExpressionTree) -> ExpressionTree:
    if et.type is NodeType.CST:
        return ExpressionTree(et.gekko, math_sqrt(et.value))
    return ExpressionTree(et.gekko, [et], NodeType.SRT)


class Equation:
    def __init__(
        self,
        lhs: ExpressionTree,
        cmp: Cmp,
        rhs: ExpressionTree,
        name: str,
        hard: bool = False,
        enforce: bool = True,
    ):
        self.lhs = lhs
        self.cmp = cmp
        self.rhs = rhs
        self.name = name
        self.hard = hard
        self.enforce = enforce

    def surplus(self) -> float:
        if self.cmp is Cmp.LE:
            return max(0.0, self.lhs.evaluate() - self.rhs.evaluate())
        elif self.cmp is Cmp.GE:
            return max(0.0, self.rhs.evaluate() - self.lhs.evaluate())
        elif self.cmp is Cmp.EQ:
            return abs(self.rhs.evaluate() - self.lhs.evaluate())

    def slack(self) -> float:
        if self.cmp is Cmp.LE:
            return max(0.0, self.rhs.evaluate() - self.lhs.evaluate())
        elif self.cmp is Cmp.GE:
            return max(0.0, self.lhs.evaluate() - self.rhs.evaluate())
        elif self.cmp is Cmp.EQ:
            return 0.0

    def apply_equation(self, gekko: GEKKO):
        if not self.enforce:
            return
        self.set_gekko(gekko)
        lhs_expr = self.lhs.get_gekko_expression(lambda x: x, None, False)
        rhs_expr = self.rhs.get_gekko_expression(lambda x: x, None, False)
        assert epsilon is not None
        e = epsilon.get_gekko_expression()
        lhs_val = self.lhs.evaluate()
        rhs_val = self.rhs.evaluate()
        e_val = epsilon.evaluate()
        if isinstance(lhs_expr, ExpressionTree) or isinstance(rhs_expr, ExpressionTree):
            raise Exception("?")
        if self.cmp is Cmp.LE:
            if self.hard:
                gekko.Equation(lhs_expr <= rhs_expr)
            else:
                gekko.Equation(lhs_expr <= rhs_expr + e)
            if lhs_val > rhs_val + e_val:
                debug(
                    "WARNING: Equation " + self.name + " is not met: ",
                    lhs_val,
                    "<=",
                    rhs_val,
                    flag=1,
                )
        elif self.cmp is Cmp.GE:
            if self.hard:
                gekko.Equation(lhs_expr >= rhs_expr)
            else:
                gekko.Equation(lhs_expr >= rhs_expr - e)
            if lhs_val < rhs_val - e_val:
                debug(
                    "WARNING: Equation " + self.name + " is not met",
                    lhs_val,
                    ">=",
                    rhs_val,
                    flag=1,
                )
        elif self.cmp is Cmp.EQ:
            if self.hard:
                gekko.Equation(lhs_expr == rhs_expr)
            else:
                gekko.Equation(lhs_expr >= rhs_expr - e)
                gekko.Equation(lhs_expr <= rhs_expr + e)
            if abs(lhs_val - rhs_val) > e_val:
                debug(
                    "WARNING: Equation " + self.name + " is not met",
                    lhs_val,
                    "==",
                    rhs_val,
                    flag=1,
                )

    def get_variable_list(self):
        return self.lhs.get_variable_list() + self.rhs.get_variable_list()

    def set_gekko(self, gekko: GEKKO):
        self.lhs.set_gekko(gekko)
        self.rhs.set_gekko(gekko)

    def is_equation_met(self):
        eps = epsilon.evaluate()
        if self.cmp is Cmp.LE:
            if self.hard:
                return self.lhs.evaluate() <= self.rhs.evaluate() + 1e-6
            else:
                return self.lhs.evaluate() <= self.rhs.evaluate() + eps + 1e-6
        elif self.cmp is Cmp.GE:
            if self.hard:
                return self.lhs.evaluate() >= self.rhs.evaluate() - 1e-6
            else:
                return self.lhs.evaluate() >= self.rhs.evaluate() - eps - 1e-6
        elif self.cmp is Cmp.EQ:
            if self.hard:
                return abs(self.lhs.evaluate() - self.rhs.evaluate()) <= 1e-6
            else:
                return (self.lhs.evaluate() >= self.rhs.evaluate() - eps - 1e-6) and (
                    self.lhs.evaluate() <= self.rhs.evaluate() + eps + 1e-6
                )
