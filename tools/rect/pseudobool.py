import collections
from typing import Callable


class Literal:
    pass


class Term:
    pass


class Expr:
    pass


AddTerm = str | int | float | Literal | Term | Expr
ROBDDTerm = int | tuple[int, int, int]


class Literal:
    def __init__(self, var: str, sign: bool = True) -> None:
        self.v = str(var)
        self.s = sign

    def __mul__(self, other: int | float):
        return Term(self, other)

    def __rmul__(self, other: int | float):
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

    def __eq__(self, other: AddTerm):
        return (Expr() + self) == (Expr() + other)

    def tostr(self):
        if not self.s:
            return "-" + self.v
        return "" + self.v


class Term:
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
    def __init__(self, c: int = 0, t: dict = None) -> None:
        if t is None:
            t = {}
        self.c = int(c)
        self.t = collections.OrderedDict()
        for v in t:
            self.t[v] = Term(t[v].L, t[v].c)

    def __add__(self, term: AddTerm):
        rets = Expr(self.c, self.t)
        if type(term) is str:
            return self + Term(Literal(term))
        elif type(term) is Literal:
            return self + Term(term)
        elif type(term) is Term:
            term: Term
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
        elif type(term) is float or type(term) is int:
            rets.c += int(term)
        elif type(term) is Expr:
            term: Expr
            rets.c += term.c
            for v in term.t:
                rets = rets + term.t[v]
        else:
            raise Exception("Invalid type")
        return rets

    def __sub__(self, term: AddTerm):
        if type(term) is str:
            return self - Term(Literal(term))
        elif type(term) is Literal:
            return self - Term(term)
        elif type(term) is Term:
            term: Term
            return self + Term(term.L, -term.c)
        elif type(term) is float or type(term) is int:
            return self + (- int(term))
        elif type(term) is Expr:
            term: Expr
            tmp = self + (-term.c)
            for v in term.t:
                tmp = tmp - term.t[v]
            return tmp
        else:
            raise Exception("Invalid type")

    def __mul__(self, term: int | float):
        if type(term) is int or type(term) is float:
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
    s = 0
    for term in lst:
        s += term.c
    return s


def largebit(n: int):
    i = 1
    while 2 * i <= n:
        i = i * 2
    return i


def insert(lst: list[Term], trm: Term):
    if trm.c == 0:
        return lst
    lst = lst + [trm]
    i = len(lst) - 1
    while i > 0 and lst[i].c > lst[i - 1].c:
        lst[i], lst[i - 1] = lst[i - 1], lst[i]
        i = i - 1
    return lst


class Ineq:
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
        self.lhs = lhs - rhs
        self.rhs = - self.lhs.c
        self.lhs.c = 0.0
        self.op = op
        self.clause = None

    def isclause(self) -> bool:
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
            i = i + 1
        s = 0
        while i < len(lst) and s <= self.rhs:
            s += lst[i].c
            i = i + 1
        if s > self.rhs or (s >= self.rhs and self.op == ">="):
            return False
        self.clause = clause
        self.clause.sort()
        return True

    def tostr(self) -> str:
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
        lst = []
        for v in self.lhs.t:
            lst.append(self.lhs.t[v])
        lst.sort(key=lambda x: -x.c)
        if self.op == ">=":
            if not coefficientdecomposition:
                return constructrobdd((lst, self.rhs),
                                      lambda x: maxsum(x[0]) < x[1] or x[1] <= 0,
                                      lambda x: 1 if x[1] <= 0 else 0,
                                      lambda x: x[0][0].L.v,
                                      lambda x: (x[0][1:], x[1] - x[0][0].c if x[0][0].L.s else x[1]),
                                      lambda x: (x[0][1:], x[1] if x[0][0].L.s else x[1] - x[0][0].c),
                                      lambda x: ",".join(map(lambda y: y.tostr(), x[0])) + ";" + str(x[1]))
            else:
                return constructrobdd((lst, self.rhs),
                                      lambda x: maxsum(x[0]) < x[1] or x[1] <= 0,
                                      lambda x: 1 if x[1] <= 0 else 0,
                                      lambda x: x[0][0].L.v,
                                      lambda x: (insert(x[0][1:], Term(x[0][0].L, x[0][0].c - largebit(x[0][0].c))),
                                                 x[1] - largebit(x[0][0].c) if x[0][0].L.s else x[1]),
                                      lambda x: (insert(x[0][1:], Term(x[0][0].L, x[0][0].c - largebit(x[0][0].c))),
                                                 x[1] if x[0][0].L.s else x[1] - largebit(x[0][0].c)),
                                      lambda x: ",".join(map(lambda y: y.tostr(), x[0])) + ";" + str(x[1]))


memory: list[ROBDDTerm] = [0, 1]
mmap = {}


def constructrobdd(data: any, bccond: Callable[[any], bool], bcconstr: Callable[[any], int],
                   dvar: Callable[[any], str | int], ifprop: Callable[[any], any], elprop: Callable[[any], any],
                   serdat: Callable[[any], str], memo: dict[str, tuple[int, int, int]] = None) -> ROBDDTerm:
    if memo is None:
        memo = {}
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
    if dv not in mmap:
        mmap[dv] = {}
    if ifnode not in mmap[dv]:
        mmap[dv][ifnode] = {}
    if elnode not in mmap[dv][ifnode]:
        obj = (dv, ifnode, elnode)
        memory.append(obj)
        mmap[dv][ifnode][elnode] = len(memory) - 1
    memo[serdat(data)] = mmap[dv][ifnode][elnode]
    return mmap[dv][ifnode][elnode]
