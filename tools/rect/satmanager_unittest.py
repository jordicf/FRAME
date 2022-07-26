import unittest

import pseudobool
import satmanager


def exactlyone(sm: satmanager.SATManager, lst: list[pseudobool.Literal]):
    e = pseudobool.Expr()
    for i in lst:
        e = e + i
    sm.heuleencoding(lst)
    sm.pseudoboolencoding(e >= 1)
    return e


def alldifferent(lst: list[int]):
    for i in lst:
        if lst.count(i) != 1:
            return False
    return True


class TestSAT(unittest.TestCase):

    def test_atmostone(self):
        sm = satmanager.SATManager()
        lst = []
        for i in range(0, 100):
            lst.append(sm.newvar(str(i)))
        e1 = exactlyone(sm, lst)
        self.assertTrue(sm.solve())
        res = sm.evalexpr(e1)
        self.assertEqual(res, 1)

    def test_sudoku(self):
        sm = satmanager.SATManager()
        table = []
        size = 3
        cord = [size * i + j for i in range(0, size) for j in range(0, size)]
        sol = []

        # Variable definition
        for i in cord:
            table.append([])
            sol.append([])
            for j in cord:
                table[i].append([])
                sol[i].append(0)
                for k in cord:
                    table[i][j].append(sm.newvar(str(i) + "." + str(j) + "." + str(k)))

        # Constraint definition
        for i in cord:
            for j in cord:
                # Every cell has exactly one number
                exactlyone(sm, [table[i][j][k] for k in cord])
                # Numbers do not repeat on each row
                exactlyone(sm, [table[i][k][j] for k in cord])
                # Numbers do not repeat on each column
                exactlyone(sm, [table[k][i][j] for k in cord])
        for i in range(0, size):
            for j in range(0, size):
                for k in cord:
                    # Numbers do not repeat on each quadrant
                    exactlyone(sm, [table[size * i + k1][size * j + k2][k] for k2 in range(0, size) for k1 in
                                    range(0, size)])

        # Solving the sudoku
        sm.solve()

        # Printing the solution
        print()
        for j1 in range(0, size):
            for j2 in range(0, size):
                for i1 in range(0, size):
                    for i2 in range(0, size):
                        for k in cord:
                            if sm.value(table[i1 * size + i2][j1 * size + j2][k]) == 1:
                                sol[i1 * size + i2][j1 * size + j2] = k + 1
                                print(sol[i1 * size + i2][j1 * size + j2], end=" ")
                    print("", end=" ")
                print("")
            print("")

        # Verifying the correctness
        for i in cord:
            # Numbers do not repeat on each row
            self.assertTrue(alldifferent([sol[i][j] for j in cord]))
            # Numbers do not repeat on each column
            self.assertTrue(alldifferent([sol[j][i] for j in cord]))
        for i in range(0, size):
            for j in range(0, size):
                # Numbers do not repeat on each quadrant
                self.assertTrue(
                    alldifferent([sol[size * i + k1][size * j + k2] for k2 in range(0, size) for k1 in range(0, size)]))


if __name__ == '__main__':
    unittest.main()
