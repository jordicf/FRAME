import collections

class Literal:
    def __init__(self, var: str, sign: bool = True) -> None:
        self.v = str(var)
        self.s = sign
    def __mul__(self, other: int):
        return Term(self, other)
    def __rmul__(self, other: int):
        return self * other
    def __neg__(self):
        return Literal(self.v, not self.s)
    def __add__(self, other):
        e = Expr() + self + other
        return e
    def __radd__(self, other):
        e = Expr() + self + other
        return e
    def __ge__(self, other):
        return (Expr() + self) >= (Expr() + other)
    def __le__(self, other):
        return (Expr() + self) <= (Expr() + other)
    def __gt__(self, other):
        return (Expr() + self) > (Expr() + other)
    def __lt__(self, other):
        return (Expr() + self) < (Expr() + other)
    def __eq__(self, other):
        return (Expr() + self) == (Expr() + other)
    def tostr(self):
        if not self.s:
            return "-" + self.v
        return "" + self.v

class Term:
    def __init__(self, lit: Literal, const: int = 1) -> None:
        self.l = Literal(lit.v, lit.s)
        self.c = int(const)
    def __mul__(self, other: int):
        return Term(self.l, self.c * int(other))
    def __rmul__(self, other: int):
        return self * other
    def __neg__(self):
        return Term(self.l, -self.c)
    def __add__(self, other):
        e = Expr() + self + other
        return e
    def __radd__(self, other):
        e = Expr() + self + other
        return e
    def __ge__(self, other):
        return (Expr() + self) >= (Expr() + other)
    def __le__(self, other):
        return (Expr() + self) <= (Expr() + other)
    def __gt__(self, other):
        return (Expr() + self) > (Expr() + other)
    def __lt__(self, other):
        return (Expr() + self) < (Expr() + other)
    def __eq__(self, other):
        return (Expr() + self) == (Expr() + other)
    def tostr(self):
        return str(self.c) + " " + self.l.tostr()

class Expr:
    def __init__(self, c: int = 0, t: dict = {}) -> None:
        self.c = int(c)
        self.t = collections.OrderedDict()
        for v in t:
            self.t[v] = Term(t[v].l, t[v].c)
    def __add__(self, term):
        rets = Expr(self.c, self.t)
        if type(term) is str:
            return self + Term( Literal(term) )
        elif type(term) is Literal:
            return self + Term( term )
        elif type(term) is Term:
            if term.c == 0.0:
                return rets
            if term.l.v in rets.t:
                if term.l.s == rets.t[term.l.v].l.s:
                    rets.t[term.l.v].c += term.c
                    if rets.t[term.l.v].c == 0:
                        del rets.t[term.l.v]
                else:
                    rets.t[term.l.v].c -= term.c
                    rets.c += term.c
                    if rets.t[term.l.v].c == 0:
                        del rets.t[term.l.v]
            else:
                rets.t[term.l.v] = Term( term.l, term.c )
            if term.l.v in rets.t and rets.t[term.l.v].c < 0:
                rets.c += rets.t[term.l.v].c
                rets.t[term.l.v].c = -rets.t[term.l.v].c
                rets.t[term.l.v].l.s = not rets.t[term.l.v].l.s
        elif type(term) is float or type(term) is int:
            rets.c += int(term)
        elif type(term) is Expr:
            rets.c += term.c
            for v in term.t:
                rets = rets + term.t[v]
        else:
            raise Exception("Invalid type")
        return rets
    def __sub__(self, term):
        if type(term) is str:
            return self - Term( Literal(term) )
        elif type(term) is Literal:
            return self - Term( term )
        elif type(term) is Term:
            return self + Term( term.l, -term.c )
        elif type(term) is float or type(term) is int:
            return self + (- int(term))
        elif type(term) is Expr:
            tmp = self + (-term.c)
            for v in term.t:
                tmp = tmp - term.t[v]
            return tmp
        else:
            raise Exception("Invalid type")
    def __mul__(self, term: int):
        if type(term) is int or type(term) is float:
            rets = Expr(self.c, self.t)
            removes = []
            for t in rets.t:
                rets.t[t].c *= int(term)
                if rets.t[t].c == 0:
                    removes.append(t)
                if t in rets.t and rets.t[t].c < 0:
                    rets.c += rets.t[t].c
                    rets.t[t] = Term( Literal(rets.t[t].l.v, not rets.t[t].l.s), -rets.t[t].c )
            for r in removes:
                del rets.t[r]
            return rets
        else:
            raise Exception("Invalid type")
    def __rmul__(self, term: int):
        return self * term
    def __ge__(self, other):
        return Ineq( self, Expr() + other, ">=")
    def __le__(self, other):
        return Ineq( self, Expr() + other, "<=")
    def __gt__(self, other):
        return Ineq( self, Expr() + other, ">")
    def __lt__(self, other):
        return Ineq( self, Expr() + other, "<")
    def __eq__(self, other):
        return Ineq( self, Expr() + other, "=")
    def tostr(self) -> str:
        res = ""
        for v in self.t:
            res += self.t[v].tostr()
            res += " + "
        res += str(self.c)
        return res

def maxsum(lst):
    s = 0
    for term in lst:
        s += term.c
    return s

def largebit(n):
    i = 1
    while 2 * i <= n:
        i = i * 2
    return i

def insert(lst, trm):
    if trm.c == 0:
        return lst
    lst = lst + [trm]
    i = len(lst) - 1
    while i > 0 and lst[i].c > lst[i-1].c:
        lst[i], lst[i-1] = lst[i-1], lst[i]
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
            return True # Tautology
        l = []
        for v in self.lhs.t:
            l.append( self.lhs.t[v] )
        l.sort(key=lambda x : -x.c)
        clause = []
        i = 0
        while i < len(l) and (l[i].c > self.rhs or (l[i].c >= self.rhs and self.op == ">=")):
            clause.append( l[i].l )
            i = i + 1
        s = 0
        while i < len(l) and s <= self.rhs:
            s += l[i].c
            i = i + 1
        if (s > self.rhs or (s >= self.rhs and self.op == ">=")):
            return False
        self.clause = clause
        self.clause.sort()
        return True
    def tostr(self) -> str:
        if self.clause != None:
            res = ""
            f = True
            for l in self.clause:
                if not f:
                    res += " + "
                f = False
                res += "1 " + l.tostr()
            return res + " >= 1"
        return self.lhs.tostr() + " " + self.op + " " + str(self.rhs)
    def getROBDD(self, coefficientDecomposition = False):
        l = []
        for v in self.lhs.t:
            l.append( self.lhs.t[v] )
        l.sort(key=lambda x : -x.c)
        if self.op == ">=":
            if not coefficientDecomposition:
                return constructROBDD( (l, self.rhs),
                              lambda x: maxsum(x[0]) < x[1] or x[1] <= 0,
                              lambda x: 1 if x[1] <= 0 else 0,
                              lambda x: x[0][0].l.v,
                              lambda x: ( x[0][1:], x[1] - x[0][0].c if x[0][0].l.s else x[1]),
                              lambda x: ( x[0][1:], x[1] if x[0][0].l.s else x[1] - x[0][0].c),
                              lambda x: ",".join(map(lambda x: x.tostr(), x[0])) + ";" + str( x[1] ) )
            else:
                return constructROBDD( (l, self.rhs),
                              lambda x: maxsum(x[0]) < x[1] or x[1] <= 0,
                              lambda x: 1 if x[1] <= 0 else 0,
                              lambda x: x[0][0].l.v,
                              lambda x: ( insert(x[0][1:], Term(x[0][0].l, x[0][0].c - largebit(x[0][0].c))),
                                          x[1] - largebit(x[0][0].c) if x[0][0].l.s else x[1]),
                              lambda x: ( insert(x[0][1:], Term(x[0][0].l, x[0][0].c - largebit(x[0][0].c))),
                                          x[1] if x[0][0].l.s else x[1] - largebit(x[0][0].c)),
                              lambda x: ",".join(map(lambda x: x.tostr(), x[0])) + ";" + str( x[1] ) )


memory = [0,1]
mmap = {}
def constructROBDD(data, bccond, bcconstr, dvar, ifprop, elprop, serdat, memo={}):
    if serdat(data) in memo:
        return memo[serdat(data)]
    if bccond(data):
        res = bcconstr(data)
        if res != 0 and res != 1:
            raise Exception("Base case has to be either 0 or 1")
        return res
    #print(serdat(data))
    ifnode = constructROBDD(ifprop(data), bccond, bcconstr, dvar, ifprop, elprop, serdat, memo)
    elnode = constructROBDD(elprop(data), bccond, bcconstr, dvar, ifprop, elprop, serdat, memo)
    if ifnode == elnode:
        return ifnode
    dv = dvar(data)
    if dv not in mmap:
        mmap[dv] = {}
    if ifnode not in mmap[dv]:
        mmap[dv][ifnode] = {}
    if elnode not in mmap[dv][ifnode]:
        obj = ( dv, ifnode, elnode )
        memory.append(obj)
        mmap[dv][ifnode][elnode] = len(memory) - 1
    memo[serdat(data)] = mmap[dv][ifnode][elnode]
    return mmap[dv][ifnode][elnode]

def reconstructROBDD(index):
    if index == 0 or index == 1:
        return index
    (v, i, e) = memory[index]
    return (v, reconstructROBDD(i), reconstructROBDD(e))
