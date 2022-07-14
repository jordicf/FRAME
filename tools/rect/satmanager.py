from pseudobool import Literal, Term, Expr, Ineq, memory
import subprocess

class SATManager:
    def __init__(self) -> None:
        self.ttable = {}
        self.tcount = 1
        self.vtable = [0]
        self.auxcount = 0
        self.clauses = []
        self.model = {}
        self.codified = {}
        self.flipped = {}
    def newVar(self, name: str, pre: str = "def_") -> Literal:
        vname = pre + str(name)
        if vname not in self.ttable:
            self.ttable[vname] = self.tcount
            self.tcount += 1
            self.vtable.append(vname)
        return Literal(vname)
    def isFlipped(self, varname):
        if varname in self.flipped and self.flipped[varname]:
            return True
        return False
    def setFlipped(self, varname, value=True):
        if value:
            self.flipped[varname] = True
        else:
            del self.flipped[varname]
    def newAux(self) -> Literal:
        self.auxcount += 1
        return self.newVar(str(self.auxcount), "aux_")
    def quadraticEncoding(self, lst) -> None:    # At most 1 constraint
        for i in range(0, len(lst)):
            for j in range(i+1, len(lst)):
                self.clauses.append( [ -lst[i], -lst[j] ] )
    def heuleEncoding(self, lst, k: int = 3) -> None: # At most 1 constraint
        if k < 3:
            raise Exception("k must be at least 3")
        if len(lst) <= k:
            self.quadraticEncoding(lst)
        else:
            fresh = self.newAux()
            h1 = lst[:k-1]
            h2 = lst[k-2:]
            h1.append(fresh)
            h2[0] = -fresh
            self.quadraticEncoding(h1)
            self.heuleEncoding(h2, k)
    def imply(self, list1, l2) -> None: # l1 -> l2
        l = list( map( lambda x: -x, list1 ) )
        l.append(l2)
        self.clauses.append(l)
    def codifyROBDD(self, robdd_id : int) -> None:
        if robdd_id not in self.codified:
            self.codified[robdd_id] = True
            pnode = self.newVar( robdd_id, "robdd_" )
            if robdd_id == 0:
                self.clauses.append( [ -pnode ] )
            elif robdd_id == 1:
                self.clauses.append( [ pnode ] )
            else:
                robdd = memory[robdd_id]
                self.codifyROBDD( robdd[1] )
                self.codifyROBDD( robdd[2] )
                dvar  = self.newVar( robdd[0], "" )
                inode = self.newVar( robdd[1], "robdd_" )
                enode = self.newVar( robdd[2], "robdd_" )
                self.clauses.append( [ -pnode, -dvar, inode ] )
                self.clauses.append( [ -pnode, dvar, enode ] )
    def pseudoboolEncoding( self, ineq: Ineq, coefficientDecomposition: bool = False ) -> None:
        if ineq.isclause():
            if ineq.clause != None:
                self.clauses.append(ineq.clause)
        else:
            robdd = ineq.getROBDD(coefficientDecomposition)
            self.codifyROBDD(robdd)
            self.clauses.append( [ self.newVar( robdd, "robdd_" ) ] )
    def printClauses(self) -> None:
        for c in self.clauses:
            s = ""
            for l in c:
                if s != "":
                    s += " v "
                s += l.tostr()
            print(s)
    def toCNF(self) -> str:
        s = "p cnf " + str(self.tcount - 1) + " " + str(len(self.clauses)) + "\n"
        for c in self.clauses:
            for l in c:
                if l.s == self.isFlipped(l.v): # not l.s
                    s += "-"
                s += str( self.ttable[l.v] ) + " "
            s += "0\n"
        return s + "\n"
    def value(self, lit: Literal) -> int:
        if lit.v not in self.model:
            return None
        if not lit.s:
            return 1 - self.model[lit.v]
        return self.model[lit.v]
    def evalExpr(self, expr: Expr) -> float:
        s = expr.c
        for t in expr.t:
            if self.value(expr.t[t].l) == None:
                return None
            if self.value(expr.t[t].l) == 1:
                s += expr.t[t].c
        return s
    def prioritize(self, lst):
        i = 1
        for l in lst:
            pos = self.ttable[l.v]
            other = self.vtable[i]
            self.ttable[other] = pos
            self.vtable[pos] = other
            self.ttable[l.v] = i
            self.vtable[i] = l.v
            self.setFlipped(l.v, not l.s)
            i = i + 1
    def solve(self) -> bool:
        out = self.toCNF()
        file = open("tmp.cnf", "w")
        file.write(out)
        file.close()
        run = subprocess.Popen(["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=32)
        run.communicate(input = b'minisat ./tmp.cnf ./output.out\n')
        run.wait()
        file = open("output.out", "r")
        fline = file.readline()
        if fline == "SAT\n":
            arr = [0] * self.tcount
            while fline:
                fline = str(file.readline())
                for word in fline.split():
                    if word[0] == '-':
                        arr[ int( word[1:] ) ] = 0
                    else:
                        arr[ int( word ) ] = 1
            for v in self.ttable:
                self.model[v] = arr[ self.ttable[v] ]
            return True
        else:
            return False
        
