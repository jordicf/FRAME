# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).


import collections
import typing

AddTerm = str | int | float | object


class Literal:
    """
    A literal is defined as either a variable (x) or its negation (not x)
    """
    def __init__(self, var: str, sign: bool = True) -> None:
        self.v = str(var)
        self.s = sign

    def __mul__(self, other: int):
        return Term(self, other)

    def __rmul__(self, other: int):
        return self * other

    def __neg__(self):
        return Literal(self.v, not self.s)

    def __add__(self, other: AddTerm):
        e = Expr() + self + other
        return e

    def __radd__(self, other: AddTerm):
        e = Expr() + self + other
        return e

    def __ge__(self, other: AddTerm):
        return (Expr() + self) >= (Expr() + other)

    def __le__(self, other: AddTerm):
        return (Expr() + self) <= (Expr() + other)

    def __gt__(self, other: AddTerm):
        return (Expr() + self) > (Expr() + other)

    def __lt__(self, other: AddTerm):
        return (Expr() + self) < (Expr() + other)

    def __eq__(self, other: object):
        return (Expr() + self) == (Expr() + other)

    def tostr(self):
        if not self.s:
            return "-" + self.v
        return "" + self.v


class Term:
    """
    A term is a literal times a constant.
    """
    def __init__(self, lit: Literal, const: int = 1) -> None:
        self.L = Literal(lit.v, lit.s)
        self.c = int(const)

    def __mul__(self, other: int | float):
        return Term(self.L, self.c * int(other))

    def __rmul__(self, other: int | float):
        return self * other

    def __neg__(self):
        return Term(self.L, -self.c)

    def __add__(self, other: AddTerm):
        e = Expr() + self + other
        return e

    def __radd__(self, other: AddTerm):
        e = Expr() + self + other
        return e

    def __ge__(self, other: AddTerm):
        return (Expr() + self) >= (Expr() + other)

    def __le__(self, other: AddTerm):
        return (Expr() + self) <= (Expr() + other)

    def __gt__(self, other: AddTerm):
        return (Expr() + self) > (Expr() + other)

    def __lt__(self, other: AddTerm):
        return (Expr() + self) < (Expr() + other)

    def __eq__(self, other: AddTerm):
        return (Expr() + self) == (Expr() + other)

    def tostr(self):
        return str(self.c) + " " + self.L.tostr()


class Expr:
    """
    A linear expression is an affine combination of literals, aka. a sum of terms plus a constant.
    No term has negative constants. Instead, -a * l is turned into -a * (1 - not l) = -a + a * not l
    """
    def __init__(self, c: int = 0, t = dict()) -> None:
        if t is None:
            t = {}
        self.c = int(c)
        self.t = collections.OrderedDict()
        for v in t:
            self.t[v] = Term(t[v].L, t[v].c)

    def __add__(self, term: AddTerm):
        rets = Expr(self.c, self.t)
        if isinstance(term, str):
            return self + Term(Literal(term))
        elif isinstance(term, Literal):
            return self + Term(term)
        elif isinstance(term, Term):
            if term.c == 0.0:
                return rets
            if term.L.v in rets.t:
                if term.L.s == rets.t[term.L.v].L.s:
                    rets.t[term.L.v].c += term.c
                    if rets.t[term.L.v].c == 0:
                        del rets.t[term.L.v]
                else:
                    rets.t[term.L.v].c -= term.c
                    rets.c += term.c
                    if rets.t[term.L.v].c == 0:
                        del rets.t[term.L.v]
            else:
                rets.t[term.L.v] = Term(term.L, term.c)
            if term.L.v in rets.t and rets.t[term.L.v].c < 0:
                rets.c += rets.t[term.L.v].c
                rets.t[term.L.v].c = -rets.t[term.L.v].c
                rets.t[term.L.v].L.s = not rets.t[term.L.v].L.s
        elif isinstance(term, float) or isinstance(term, int):
            rets.c += int(term)
        elif isinstance(term, Expr):
            rets.c += term.c
            for v in term.t:
                rets = rets + term.t[v]
        else:
            raise Exception("Invalid type")
        return rets

    def __sub__(self, term: AddTerm):
        if isinstance(term, str):
            return self - Term(Literal(term))
        elif isinstance(term, Literal):
            return self - Term(term)
        elif isinstance(term, Term):
            return self + Term(term.L, -term.c)
        elif isinstance(term, float) or isinstance(term, int):
            return self + (- int(term))
        elif isinstance(term, Expr):
            tmp = self + (-term.c)
            for v in term.t:
                tmp = tmp - term.t[v]
            return tmp
        else:
            raise Exception("Invalid type")

    def __mul__(self, term: int | float):
        if isinstance(term, int) or isinstance(term, float):
            rets = Expr(self.c, self.t)
            removes = []
            for t in rets.t:
                rets.t[t].c *= int(term)
                if rets.t[t].c == 0:
                    removes.append(t)
                if t in rets.t and rets.t[t].c < 0:
                    rets.c += rets.t[t].c
                    rets.t[t] = Term(Literal(rets.t[t].L.v, not rets.t[t].L.s), -rets.t[t].c)
            for r in removes:
                del rets.t[r]
            return rets
        else:
            raise Exception("Invalid type")

    def __rmul__(self, term: int | float):
        return self * term

    def __ge__(self, other: AddTerm):
        return Ineq(self, Expr() + other, ">=")

    def __le__(self, other: AddTerm):
        return Ineq(self, Expr() + other, "<=")

    def __gt__(self, other: AddTerm):
        return Ineq(self, Expr() + other, ">")

    def __lt__(self, other: AddTerm):
        return Ineq(self, Expr() + other, "<")

    def __eq__(self, other: AddTerm):
        return Ineq(self, Expr() + other, "=")

    def tostr(self) -> str:
        res = ""
        for v in self.t:
            res += self.t[v].tostr()
            res += " + "
        res += str(self.c)
        return res


def maxsum(lst: list[Term]):
    """
    Returns the maximum value an expression (or, rather, a list of terms) can have
    Assumes no term has negative constants
    :param lst: The expression
    """
    s = 0
    for term in lst:
        s += term.c
    return s


def largebit(n: int):
    """
    Given a number n, returns the largest 2**k such that 2**k < n.
    :param n: The number.
    """
    i = 1
    while 2 * i <= n:
        i *= 2
    return i


def insert(lst: list[Term], trm: Term):
    """
    Inserts the term into the list, and guarantees sorting by constant term
    :param lst: List of terms
    :param trm: New term to add
    """
    if trm.c == 0:
        return lst
    lst.append(trm)
    i = len(lst) - 1
    while i > 0 and lst[i].c > lst[i - 1].c:
        lst[i], lst[i - 1] = lst[i - 1], lst[i]
        i -= 1
    return lst


BaseType = tuple[list[Term], int]


class Ineq:
    """
    A linear inequality is something of the form A >= B, where A and B are linear expressions.
    """
    def __init__(self, lhs: Expr = Expr(), rhs: Expr = Expr(), op: str = ">=") -> None:
        if op != ">=" and op != "<=" and op != ">" and op != "<" and op != "=" and op != "==":
            raise Exception("Invalid operator")
        if op == "<=":
            lhs, rhs = rhs, lhs
            op = ">="
        if op == "<":
            lhs, rhs = rhs, lhs
            op = ">"
        if op == "==":
            op = "="
        self.lhs: Expr = lhs - rhs
        self.rhs: int = - self.lhs.c
        self.lhs.c = 0
        self.op: str = op
        self.clause: None | list[Literal] = None

    def isclause(self) -> bool:
        """
        Returns true if the inequality can be expressed as l1 v l2 v ...,
        where l1, l2... are literals.
        """
        if self.op != ">=" and self.op != ">":
            return False
        if self.rhs <= 0:
            return True  # Tautology
        lst = []
        for v in self.lhs.t:
            lst.append(self.lhs.t[v])
        lst.sort(key=lambda x: -x.c)
        clause = []
        i = 0
        while i < len(lst) and (lst[i].c > self.rhs or (lst[i].c >= self.rhs and self.op == ">=")):
            clause.append(lst[i].L)
            i += 1
        s = 0
        while i < len(lst) and s <= self.rhs:
            s += lst[i].c
            i += 1
        if s > self.rhs or (s >= self.rhs and self.op == ">="):
            return False
        self.clause = clause
        self.clause.sort()
        return True

    def tostr(self) -> str:
        """
        Turns the constraint into a human-readable string.
        """
        if self.clause is not None:
            res = ""
            f = True
            for L in self.clause:
                if not f:
                    res += " + "
                f = False
                res += "1 " + L.tostr()
            return res + " >= 1"
        return self.lhs.tostr() + " " + self.op + " " + str(self.rhs)

    def getrobdd(self, coefficientdecomposition: bool = False) -> int:
        """
        Turns the constraint into a ROBDD
        :param coefficientdecomposition: This would guarantee the size of the ROBDD to be polynomial (n^2), but I'd
        argue against it most of the time, since tends to produce larger graphs most of the time, and also it does
        not guarantee arc consistency by unit propagation, which is bad
        """
        lst = []
        for v in self.lhs.t:
            lst.append(self.lhs.t[v])
        lst.sort(key=lambda x: -x.c)

        data: BaseType
        bccond: typing.Callable[[BaseType], bool]
        bcconstr: typing.Callable[[BaseType], int]
        dvar: typing.Callable[[BaseType], str]
        ifprop: typing.Callable[[BaseType], BaseType]
        elprop: typing.Callable[[BaseType], BaseType]
        serdat: typing.Callable[[BaseType], str]
        if self.op == ">=":
            if not coefficientdecomposition:
                data = (lst, self.rhs)

                def bccond(x):
                    return maxsum(x[0]) < x[1] or x[1] <= 0

                def bcconstr(x):
                    return 1 if x[1] <= 0 else 0

                def dvar(x):
                    return x[0][0].L.v

                def ifprop(x):
                    return x[0][1:], x[1] - x[0][0].c if x[0][0].L.s else x[1]

                def elprop(x):
                    return x[0][1:], x[1] if x[0][0].L.s else x[1] - x[0][0].c

                def serdat(x):
                    return ",".join(map(lambda y: y.tostr(), x[0])) + ";" + str(x[1])

                return constructrobdd(data, bccond, bcconstr, dvar, ifprop, elprop, serdat)
            else:
                data = (lst, self.rhs)

                def bccond(x):
                    return maxsum(x[0]) < x[1] or x[1] <= 0

                def bcconstr(x):
                    return 1 if x[1] <= 0 else 0

                def dvar(x):
                    return x[0][0].L.v

                def ifprop(x):
                    return (insert(x[0][1:], Term(x[0][0].L, x[0][0].c - largebit(x[0][0].c))),
                            x[1] - largebit(x[0][0].c) if x[0][0].L.s else x[1])

                def elprop(x):
                    return (insert(x[0][1:], Term(x[0][0].L, x[0][0].c - largebit(x[0][0].c))),
                            x[1] if x[0][0].L.s else x[1] - largebit(x[0][0].c))

                def serdat(x):
                    return ",".join(map(lambda y: y.tostr(), x[0])) + ";" + str(x[1])

                return constructrobdd(data, bccond, bcconstr, dvar, ifprop, elprop, serdat)
        else:
            raise Exception("Not implemented yet.")


"""
memory: Every ROBDD generated so far.
mmap:   Given a decision variable and the two branches, tells you whether this ROBDD already exists or not, and returns
        the given ROBDD if it exists. Important to guarantee canonicity.
"""
memory: list[int | tuple[(str | int), int, int]] = [0, 1]
mmap: dict[int | tuple[(str | int), int, int], int] = {}

T = typing.TypeVar('T')


def constructrobdd(data: typing.Any, bccond: typing.Callable[[T], bool], bcconstr: typing.Callable[[T], int],
                   dvar: typing.Callable[[T], str | int], ifprop: typing.Callable[[T], T],
                   elprop: typing.Callable[[T], T], serdat: typing.Callable[[T], str],
                   memo = dict[str, int]()) -> int:
    """
    Generates the ROBDD that represents the given binary function
    :param data: The data yet to be converted into an ROBDD
    :param bccond: Short for "Base Case Condition". Returns true if the data meets the condition for the base case
    :param bcconstr: Short for "Base Case Construction". Returns 0 or 1, depending on the base case to be applied
    :param dvar: Short for "Decision Variable". Returns the top decision variable of the ROBDD to construct
    :param ifprop: Short for "if propagation". Transforms the data for the if branch of the ROBDD
    :param elprop: Short for "else propagation". Transforms the data for the else branch of the ROBDD
    :param serdat: Short for "serialize data". Turns the arbitrary input type into a string - Important for efficiency
    :param memo: Short for "memoization". Remembers previous results for greater efficiency
    """
    
    if serdat(data) in memo:
        return memo[serdat(data)]
    if bccond(data):
        res = bcconstr(data)
        if res != 0 and res != 1:
            raise Exception("Base case has to be either 0 or 1")
        return res
    # print(serdat(data))
    ifnode = constructrobdd(ifprop(data), bccond, bcconstr, dvar, ifprop, elprop, serdat, memo)
    elnode = constructrobdd(elprop(data), bccond, bcconstr, dvar, ifprop, elprop, serdat, memo)
    if ifnode == elnode:
        return ifnode
    dv = dvar(data)
    obj = (dv, ifnode, elnode)
    if obj not in mmap:
        memory.append(obj)
        mmap[obj] = len(memory) - 1
    memo[serdat(data)] = mmap[obj]
    return mmap[obj]
