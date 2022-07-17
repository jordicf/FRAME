import unittest

import tools.rect.pseudobool

pseudobool = tools.rect.pseudobool


class TestPseudoboolean(unittest.TestCase):

    def test_expr1(self):
        a = pseudobool.Literal('a')
        b = pseudobool.Literal('b')
        c = pseudobool.Literal('c')
        e1 = pseudobool.Expr() + 5 * a + 3 * b + 7 * c + 12
        e2 = pseudobool.Expr() - 5 * (-a) + 2 * b + 7 * c + 1 * b + 17
        self.assertEqual(e1.tostr(), e2.tostr())

    def test_expr2(self):
        e1 = pseudobool.Expr()
        e2 = pseudobool.Expr()
        for i in range(0, 100):
            v = pseudobool.Literal('v' + str(i))
            e1 = e1 + v
            e2 = e2 + 5 * v
        self.assertEqual((5 * e1).tostr(), e2.tostr())

    def test_ineq(self):
        a = pseudobool.Literal('a')
        b = pseudobool.Literal('b')
        c = pseudobool.Literal('c')
        i = a + 4 * b + 5 * c >= 3
        self.assertEqual(i.tostr(), '1 a + 4 b + 5 c + 0.0 >= 3')
        self.assertTrue(i.isclause())
        self.assertEqual(i.tostr(), '1 b + 1 c >= 1')


if __name__ == '__main__':
    unittest.main()
