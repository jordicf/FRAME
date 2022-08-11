import subprocess

from tools.rect.pseudobool import Literal, Expr, Ineq, memory


class SATManager:
    def __init__(self) -> None:
        self.ttable:   dict[str, int] = {}
        self.tcount:   int = 1
        self.vtable:   list[str] = ['0']
        self.auxcount: int = 0
        self.clauses:  list[list[Literal]] = []
        self.model:    dict[str, int] = {}
        self.codified: dict[int, bool] = {}
        self.flipped:  dict[str, bool] = {}

    def newvar(self, name: int | float | str, pre: str = "def_") -> Literal:
        vname = pre + str(name)
        if vname not in self.ttable:
            self.ttable[vname] = self.tcount
            self.tcount += 1
            self.vtable.append(vname)
        return Literal(vname)

    def isflipped(self, varname: str) -> bool:
        if varname in self.flipped and self.flipped[varname]:
            return True
        return False

    def setflipped(self, varname: str, value: bool = True) -> None:
        if value:
            self.flipped[varname] = True
        else:
            del self.flipped[varname]

    def newaux(self) -> Literal:
        self.auxcount += 1
        return self.newvar(str(self.auxcount), "aux_")

    def quadraticencoding(self, lst: list[Literal]) -> None:  # At most 1 constraint
        for i in range(0, len(lst)):
            for j in range(i + 1, len(lst)):
                self.clauses.append([-lst[i], -lst[j]])

    def heuleencoding(self, lst: list[Literal], k: int = 3) -> None:  # At most 1 constraint
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
        lst = list(map(lambda x: -x, list1))
        lst.append(l2)
        self.clauses.append(lst)

    def _codifyrobdd(self, robdd_id: int) -> None:
        if robdd_id not in self.codified:
            self.codified[robdd_id] = True
            pnode = self.newvar(robdd_id, "robdd_")
            if robdd_id == 0:
                self.clauses.append([-pnode])
            elif robdd_id == 1:
                self.clauses.append([pnode])
            else:
                robdd = memory[robdd_id]
                self._codifyrobdd(robdd[1])
                self._codifyrobdd(robdd[2])
                dvar = self.newvar(robdd[0], "")
                inode = self.newvar(robdd[1], "robdd_")
                enode = self.newvar(robdd[2], "robdd_")
                self.clauses.append([-pnode, -dvar, inode])
                self.clauses.append([-pnode, dvar, enode])

    def pseudoboolencoding(self, ineq: Ineq, coefficientdecomposition: bool = False) -> None:
        if ineq.isclause():
            if ineq.clause is not None:
                self.clauses.append(ineq.clause)
        else:
            robdd = ineq.getrobdd(coefficientdecomposition)
            self._codifyrobdd(robdd)
            self.clauses.append([self.newvar(robdd, "robdd_")])

    def printclauses(self) -> None:
        for c in self.clauses:
            s = ""
            for lit in c:
                if s != "":
                    s += " v "
                s += lit.tostr()
            print(s)

    def tocnf(self) -> str:
        s = "p cnf " + str(self.tcount - 1) + " " + str(len(self.clauses)) + "\n"
        for c in self.clauses:
            for lit in c:
                if lit.s == self.isflipped(lit.v):  # not lit.s
                    s += "-"
                s += str(self.ttable[lit.v]) + " "
            s += "0\n"
        return s + "\n"

    def value(self, lit: Literal) -> int | None:
        if lit.v not in self.model:
            return None
        if not lit.s:
            return 1 - self.model[lit.v]
        return self.model[lit.v]

    def evalexpr(self, expr: Expr) -> float | None:
        s = expr.c
        for t in expr.t:
            if self.value(expr.t[t].L) is None:
                return None
            if self.value(expr.t[t].L) == 1:
                s += expr.t[t].c
        return s

    def prioritize(self, lst: list[Literal]) -> None:
        i = 1
        for lit in lst:
            pos = self.ttable[lit.v]
            other = self.vtable[i]
            self.ttable[other] = pos
            self.vtable[pos] = other
            self.ttable[lit.v] = i
            self.vtable[i] = lit.v
            self.setflipped(lit.v, not lit.s)
            i = i + 1

    def solve(self) -> bool:
        out = self.tocnf()
        file = open("tmp.cnf", "w")
        file.write(out)
        file.close()
        run = subprocess.Popen(["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=32)
        run.communicate(input=b'minisat ./tmp.cnf ./output.out\n')
        run.wait()
        file = open("output.out", "r")
        fline = file.readline()
        if fline == "SAT\n":
            arr = [0] * self.tcount
            while fline:
                fline = str(file.readline())
                for word in fline.split():
                    if word[0] == '-':
                        arr[int(word[1:])] = 0
                    else:
                        arr[int(word)] = 1
            for v in self.ttable:
                self.model[v] = arr[self.ttable[v]]
            file.close()
            return True
        else:
            file.close()
            return False
