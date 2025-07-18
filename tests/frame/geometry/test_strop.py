"""
Testing the stop module.
A set of matrices are given, with their corresponding STROP instances.
The test checks that all these solutions are generated.
"""

import unittest
from frame.geometry.strop import Strop, str2BoolMatrix


matrix1 = """
001100
011110
111111
001111
001010"""

sol1_1 = """\
  22
 1223
000000
  4567
  4 6"""

sol1_2 = """\
  11
 5000
660004
  0004
  2 3"""

##################################

matrix2 = """
00011011
01111110
01111110
00011111
11111111
01111111
01111100
01111100
01100100
01100000"""

#########################################
matrix3 = """
0001000
0011100
0111110
1111111
0111110
0011100
0001000
"""

sol3_1 = """\
   0
  601
 77022
8880333
 99044
  a05
   0"""

sol3_2 = """\
   2
  123
 00000
8000007
 00000
  456
   5"""

sol3_3 = """\
   3
  234
 12345
0000000
 6789a
  789
   8"""

sol3_4 = """\
   1
  000
 60003
7700044
 80005
  000
   2"""

######################################################
matrix4 = """
101
011
111
"""
#######################################################

str_matrix1 = "1 0.9 -1 0 0.2 -1 0.5 0"
str_matrix2 = "0 0 0.2 -1 1 0 0.3"
str_matrix3 = "0 0.4 -1 0.6 1.1"
str_matrix4 = "0 0.4 -1 0.6 1.0 -1"
str_matrix5 = "1.0 0.3 0 -1 0 0.2 -1 1 0.5 0.6 0.5"

bool_matrix1 = [[True, True], [False, True], [True, False]]
bool_matrix2 = [[False, False, True], [True, False, True]]


def check_matrix(matrix: str, sol: set[str] = set()) -> bool:
    """Check that the matrix generates the set of solutions."""
    s = Strop(matrix)
    for tree in s.instances():
        str_tree = str(tree)
        if str_tree not in sol:
            return False
        sol.remove(str_tree)
    return len(sol) == 0


class TestSTROP(unittest.TestCase):
    def test_check_matrices_and_solutions(self):
        self.assertTrue(check_matrix(matrix1, {sol1_1, sol1_2}))
        self.assertTrue(check_matrix(matrix2))
        self.assertTrue(check_matrix(matrix3, {sol3_1, sol3_2, sol3_3, sol3_4}))
        self.assertTrue(check_matrix(matrix4))

    def test_str2boolmatrix(self):
        self.assertEqual(str2BoolMatrix(str_matrix1), bool_matrix1)
        self.assertEqual(str2BoolMatrix(str_matrix2), bool_matrix2)
        self.assertRaises(AssertionError, str2BoolMatrix, str_matrix3)
        self.assertRaises(AssertionError, str2BoolMatrix, str_matrix4)
        self.assertRaises(AssertionError, str2BoolMatrix, str_matrix5)


if __name__ == "__main__":
    unittest.main()
