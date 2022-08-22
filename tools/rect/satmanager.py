from tools.rect.pseudobool import Literal, Expr, Ineq, memory
from pysat.solvers import Solver


Clause = list[Literal]


class SATManager:
    """
    The SATManager class implements many useful functions for the purpose of modeling using SAT
    and provides an interface with the SAT Solver.
    solver: The SAT Solver object
    ttable: Short for "Translation table". Translates variable names (str) into their names in the inner representation
    tcount: Short for "Translation count", counts the number of variables
    vtable: Short for "Variable table", lists every variable name (str) in order
    auxcount: The number of auxiliary variables set
    clauses: List of clauses set. A clause in CNF is a disjunction of literals - analog to the cube in DNF
    model: The values the SAT Solver has assigned to each variable. If no call to the SAT Solver was done, it is empty
    codified: Determines whether the constraint that variable tries to implement has already been codified or not
    flipped: Determines whether the variable should be flipped in the CNF or not (not very useful)
    """
    def __init__(self) -> None:
        self.solver:   Solver = Solver()
        self.ttable:   dict[str, int] = {}
        self.tcount:   int = 1
        self.vtable:   list[str] = ['0']
        self.auxcount: int = 0
        self.clauses:  list[Clause] = []
        self.model:    dict[str, int] = {}
        self.codified: dict[int, bool] = {}
        self.flipped:  dict[str, bool] = {}

    def add_clause(self, clause: Clause) -> None:
        """
        Adds a clause to the model
        :param clause: The clause to add
        """
        self.clauses.append(clause)

    def newvar(self, name: int | float | str, pre: str = "def_") -> Literal:
        """
        Defines a variable for the model - If the variable already exists, returns that variable instead
        :param name: The name of the variable
        :param pre: The prefix of the variable name. Useful for avoiding collisions
        """
        vname = pre + str(name)
        if vname not in self.ttable:
            self.ttable[vname] = self.tcount
            self.tcount += 1
            self.vtable.append(vname)
        return Literal(vname)

    def isflipped(self, varname: str) -> bool:
        """
        Determines whether a variable has been flipped or not
        :param varname: The name of the variable to check
        """
        if varname in self.flipped and self.flipped[varname]:
            return True
        return False

    def setflipped(self, varname: str, value: bool = True) -> None:
        """
        Flips or unflips a variable
        :param varname: Name of the variable to flip
        :param value: Whether to flip the variable or not
        """
        if value:
            self.flipped[varname] = True
        else:
            del self.flipped[varname]

    def newaux(self) -> Literal:
        """
        Returns an auxiliary variable.
        """
        self.auxcount += 1
        return self.newvar(str(self.auxcount), "aux_")

    def quadraticencoding(self, lst: list[Literal]) -> None:  # At most 1 constraint
        """
        Encodes the "at most one" cardinality constraint of the list of literals, using a quadratic approach
        :param lst: The list of literals
        """
        for i in range(0, len(lst)):
            for j in range(i + 1, len(lst)):
                self.add_clause([-lst[i], -lst[j]])

    def heuleencoding(self, lst: list[Literal], k: int = 3) -> None:  # At most 1 constraint
        """
        Encodes the "at most one" cardinality constraint of the list of literals, using Heule's linear encoding
        :param lst: The list of literals
        :param k: The size of each sublist
        """
        if k < 3:
            raise Exception("k must be at least 3")
        if len(lst) <= k:
            self.quadraticencoding(lst)
        else:
            fresh = self.newaux()
            h1 = lst[:k - 1]
            h2 = lst[k - 2:]
            h1.append(fresh)
            h2[0] = -fresh
            self.quadraticencoding(h1)
            self.heuleencoding(h2, k)

    def imply(self, list1: list[Literal], l2: Literal) -> None:  # l1 -> l2
        """
        Adds a clause (a and b and c and ...) -> x
        :param list1: The literals a, b, c... in the previous description
        :param l2: The literal x in the previous description
        """
        lst = list(map(lambda x: -x, list1))
        lst.append(l2)
        self.add_clause(lst)

    def _codifyrobdd(self, robdd_id: int) -> None:
        """
        Auxiliary function that codifies a ROBDD into SAT
        :robdd_id: The ID of the ROBDD to codify (a unique identifier)
        """
        if robdd_id not in self.codified:
            self.codified[robdd_id] = True
            pnode = self.newvar(robdd_id, "robdd_")
            if robdd_id == 0:
                self.add_clause([-pnode])
            elif robdd_id == 1:
                self.add_clause([pnode])
            else:
                robdd = memory[robdd_id]
                if isinstance(robdd, tuple):
                    self._codifyrobdd(robdd[1])
                    self._codifyrobdd(robdd[2])
                    dvar = self.newvar(robdd[0], "")
                    inode = self.newvar(robdd[1], "robdd_")
                    enode = self.newvar(robdd[2], "robdd_")
                    self.add_clause([-pnode, -dvar, inode])
                    self.add_clause([-pnode, dvar, enode])
                else:
                    raise Exception("ROBDD index " + str(robdd_id) + " should be a tuple but it's not!")

    def pseudoboolencoding(self, ineq: Ineq, coefficientdecomposition: bool = False) -> None:
        """
        Codifies a pseudoboolean inequality using ROBDDs
        :param ineq: The inequation to implement
        :param coefficientdecomposition: This would guarantee the size of the ROBDD to be polynomial (n^2), but I'd
        argue against it most of the time, since tends to produce larger graphs most of the time, and also it does
        not guarantee arc consistency by unit propagation, which is bad
        """
        if ineq.isclause():
            if ineq.clause is not None:
                self.add_clause(ineq.clause)
        else:
            robdd = ineq.getrobdd(coefficientdecomposition)
            self._codifyrobdd(robdd)
            self.add_clause([self.newvar(robdd, "robdd_")])

    def printclauses(self) -> None:
        """
        Prints all the clauses in a human-readable format.
        """
        for c in self.clauses:
            s = ""
            for lit in c:
                if s != "":
                    s += " v "
                s += lit.tostr()
            print(s)

    def tocnf(self) -> str:
        """
        Outputs a .cnf file that solves the problem (deprecated)
        """
        s = "p cnf " + str(self.tcount - 1) + " " + str(len(self.clauses)) + "\n"
        for c in self.clauses:
            for lit in c:
                if lit.s == self.isflipped(lit.v):  # not lit.s
                    s += "-"
                s += str(self.ttable[lit.v]) + " "
            s += "0\n"
        return s + "\n"

    def value(self, lit: Literal) -> int | None:
        """
        Tells you the value of a literal in the current model.
        If the SAT Solver was never called, returns None.
        :param lit: The literal to be checked.
        """
        if lit.v not in self.model:
            return None
        if not lit.s:
            return 1 - self.model[lit.v]
        return self.model[lit.v]

    def evalexpr(self, expr: Expr) -> float | None:
        """
        Evaluates the expression using the current model.
        If the SAT Solver was never called, returns None.
        :param expr: The expression to be evaluated.
        """
        s = expr.c
        for t in expr.t:
            if self.value(expr.t[t].L) is None:
                return None
            if self.value(expr.t[t].L) == 1:
                s += expr.t[t].c
        return s

    def prioritize(self, lst: list[Literal]) -> None:
        """
        Makes a list of variables have a lower id (deprecated)
        :param lst: The list of literals to prioritize
        """
        i = 1
        for lit in lst:
            pos = self.ttable[lit.v]
            other = self.vtable[i]
            self.ttable[other] = pos
            self.vtable[pos] = other
            self.ttable[lit.v] = i
            self.vtable[i] = lit.v
            self.setflipped(lit.v, not lit.s)
            i += 1

    def solve(self) -> bool:
        """
        Calls the SAT Solver and gets a solution to the current model, if such exists.
        Returns whether a solution was found.
        """
        # Wipe the solver clean
        self.solver = Solver()

        # Define the model again
        for clause in self.clauses:
            c: list[int] = list(map(
                lambda x: - self.ttable[x.v] if x.s == self.isflipped(x.v) else self.ttable[x.v], clause))
            self.solver.add_clause(c)

        # Solve
        if self.solver.solve():
            arr = [0] * self.tcount
            mod = self.solver.get_model()
            for i in range(0, len(mod)):
                word = mod[i]
                if word < 0:
                    arr[-word] = 0
                else:
                    arr[word] = 1
            for v in self.ttable:
                self.model[v] = arr[self.ttable[v]]
            return True
        else:
            return False
