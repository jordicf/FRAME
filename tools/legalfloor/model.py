from __future__ import annotations
from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Value
from math import sqrt as math_sqrt
from typing import Any, Callable
from enum import IntEnum
from tools.legalfloor.expression_tree import Equation, ExpressionTree, NodeType, Cmp, set_epsilon_gekko, sqrt


class ModelWrapper:
    def __init__(self, gekko: GEKKO, wl_mult: float = 1):
        self.gekko: GEKKO = gekko
        self.constraints: dict[str, list[Equation]] = dict()
        self.macro_constraints: dict[str, list[Equation]] = dict()
        self.objective: ExpressionTree = ExpressionTree(gekko, 0)
        self.fixed_vars: list[tuple[ExpressionTree, float]] = []
        self.variable_list: list[ExpressionTree] = []
        self.variable_set: set[str] = set()
        self.dif_cost = 0.00001
        self.coordinates: list[tuple[ExpressionTree, ExpressionTree, ExpressionTree, ExpressionTree]] = []
        self.wl_mult = wl_mult
        self.macros = []

    def add_macro(self, macro):
        self.macros.append(macro)

    def add_coordinates(self, x: ExpressionTree, y: ExpressionTree, w: ExpressionTree, h: ExpressionTree):
        self.coordinates.append((x, y, w, h))

    # def dif_cost_objective(self):
    #     obj = None
    #     for var in self.variable_list:
    #         if obj is None:
    #             obj = sqrt((var - var.evaluate())**2 + 0.0001)
    #         else:
    #             obj += sqrt((var - var.evaluate()) ** 2 + 0.0001)
    #     if obj is None:
    #         return 0
    #     return obj * self.dif_cost

    def dif_cost_objective(self):
        cost_terms = []  # 存储中间变量项
        for var in self.variable_list:
            # 为每个变量创建中间表达式
            delta_sq = self.gekko.Intermediate((var - var.evaluate())**2 + 0.0001)
            term = self.gekko.Intermediate(self.gekko.sqrt(delta_sq))
            cost_terms.append(term)
        
        # 使用gekko的sum函数进行高效求和
        total_cost = self.gekko.sum(cost_terms) * self.dif_cost
        return total_cost

    def add_constraint(self, group: str, eq: Equation):
        if group not in self.constraints:
            self.constraints[group] = []
        self.constraints[group].append(eq)
        for var in eq.get_variable_list():
            if var.data['name'] not in self.variable_set:
                self.variable_set.add(var.data['name'])
                self.variable_list.append(var)
        return eq

    def set_objective_function(self, objective: ExpressionTree):
        self.objective = objective
        for var in objective.get_variable_list():
            if var.data['name'] not in self.variable_set:
                self.variable_set.add(var.data['name'])
                self.variable_list.append(var)

    def fix_variable(self, var: ExpressionTree, val: ExpressionTree, name: str):
        assert var.type == NodeType.VAR
        self.fixed_vars.append((var, val.evaluate()))
        self.add_constraint("Fix", Equation(var, Cmp.EQ, val, name))

    def build_model(self, small_steps=False, radius=None):
        self.gekko = GEKKO(remote=False)
        self.gekko.options.SOLVER = 3
        self.gekko.options.IMODE = 2
        self.gekko.options.COLDSTART = 1
        self.gekko.options.MAX_ITER = 1
        self.gekko.MEAS_CHK = 0
        self.gekko.options.REDUCE = 3  
        set_epsilon_gekko(self.gekko)
        self.macro_constraints = dict()
        for macro in self.macros:
            for namespace, equation in macro.get_constraints(self):
                if namespace not in self.macro_constraints:
                    self.macro_constraints[namespace] = []
                self.macro_constraints[namespace].append(equation)
        if small_steps:
            assert isinstance(radius, float) or isinstance(radius, int)
            self.force_step(radius * 0.3)
        for group in self.constraints:
            for constraint in self.constraints[group]:
                constraint.apply_equation(self.gekko)
        for group in self.macro_constraints:
            for constraint in self.macro_constraints[group]:
                constraint.apply_equation(self.gekko)
        self.objective.set_gekko(self.gekko)
        if small_steps:
            self.gekko.Obj(self.objective.get_gekko_expression() * self.wl_mult)
        else:
            self.gekko.Obj(self.objective.get_gekko_expression() * self.wl_mult + self.dif_cost_objective())

    def force_step(self, radius: float):
        eq_per_coord = 6
        if 'radius' not in self.constraints:
            self.constraints['radius'] = []
            for i in range(0, eq_per_coord*len(self.coordinates)):
                self.constraints['radius'].append(Equation(0, Cmp.EQ, 1, "NULL_EQ"))
        for i, (x, y, w, h) in enumerate(self.coordinates):
            j = eq_per_coord*i
            x0 = x.evaluate()
            y0 = y.evaluate()
            w0 = w.evaluate()
            h0 = h.evaluate()
            self.constraints['radius'][j] = Equation(x, Cmp.LE, ExpressionTree(self.gekko, x0 + radius), "Cap_x0[%i]" % i, hard=True)
            self.constraints['radius'][j+1] = Equation(x, Cmp.GE, ExpressionTree(self.gekko, x0 - radius), "Cap_x1[%i]" % i, hard=True)
            self.constraints['radius'][j+2] = Equation(y, Cmp.LE, ExpressionTree(self.gekko, y0 + radius), "Cap_y0[%i]" % i, hard=True)
            self.constraints['radius'][j+3] = Equation(y, Cmp.GE, ExpressionTree(self.gekko, y0 - radius), "Cap_y1[%i]" % i, hard=True)
            self.constraints['radius'][j+4] = Equation(w, Cmp.LE, ExpressionTree(self.gekko, w0 + radius), "Cap_w[%i]" % i, hard=True)
            self.constraints['radius'][j+5] = Equation(h, Cmp.LE, ExpressionTree(self.gekko, h0 + radius), "Cap_h[%i]" % i, hard=True)

    def solve(self, verbose=False, small_steps=False, radius=None, otol=None, rtol=None) -> None:
        self.turn_off_rects(0.1)
        self.fuse_rects()
        self.build_model(small_steps=small_steps, radius=radius)
        self.gekko.options.COLDSTART = 1
        self.gekko.options.OTOL = otol
        self.gekko.options.RTOL = rtol
        self.gekko.options.MAX_ITER += 50000
        self.gekko.solve(disp=verbose, debug=0)
        for (x, v) in self.fixed_vars:
            x.value.value = [v]

    def turn_off_rects(self, area: float):
        for m in self.macros:
            m.turn_off_rects(area)

    def fuse_rects(self, perc: float = 0.05):
        for m in self.macros:
            m.fuse_rects(perc)

    def total_surplus(self) -> float:
        surplus = 0
        for category in self.constraints:
            for constraint in self.constraints[category]:
                surplus += constraint.surplus()
        return surplus

    def fix_as_lower_bound(self, expr: ExpressionTree):
        if 'lower_bound' not in expr.unary_equations:
            eq = Equation(expr, Cmp.GE, ExpressionTree(self.gekko, expr.evaluate()), "lower_bound", hard=True)
            self.add_constraint("Lower Bound", eq)
            expr.unary_equations['lower_bound'] = eq
        else:
            expr.unary_equations['lower_bound'].rhs = ExpressionTree(self.gekko, expr.evaluate())

    def fix(self, expr: ExpressionTree):
        if 'exact_value' not in expr.unary_equations:
            eq = Equation(expr, Cmp.EQ, ExpressionTree(self.gekko, expr.evaluate()), "exact_value", hard=True)
            self.add_constraint("Exact Value", eq)
            expr.unary_equations['exact_value'] = eq
        else:
            expr.unary_equations['exact_value'].rhs = ExpressionTree(self.gekko, expr.evaluate())

    def undo(self):
        for var in self.variable_list:
            var.undo()

    def verify(self, enforcer: set[str], do_undo=True, verbose=False) -> set[str]:
        done_undo = False
        for group in self.constraints:
            print_group_header = False
            for constraint in self.constraints[group]:
                if not constraint.is_equation_met():
                    if not print_group_header:
                        print("[%s]" % group)
                        print_group_header = True
                    if group == 'Inter' and not constraint.enforce:
                        print("\t[!] Equation %s not met [Far]" % constraint.name)
                        enforcer.add(constraint.name)
                        if do_undo and not done_undo:
                            done_undo = True
                            self.undo()
                    else:
                        print("\t[!] Equation %s not met" % constraint.name)

                elif verbose:
                    if not print_group_header:
                        print("[%s]" % group)
                        print_group_header = True
                    print("\t[ ] Equation %s met" % constraint.name)
        if not done_undo:
            return set()
        return enforcer