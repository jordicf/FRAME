"""
Testing the stop module.
A set of matrices are given, with their corresponding STROP instances.
The test checks that all these solutions are generated.
"""

import unittest
import numpy as np
from frame.geometry.strop import Polygon, OccMatrix


def str2OccMatrix(s: str) -> OccMatrix:
    """It generates an occupancy matrix from a string. The string constains
    the number of rows of columns of the matrix followed by a sequence
    of floats in [0,1] representing the occupancy of each cell.
    An exception is raised in case the string does not represent
    a valid matrix."""

    err_msg = "Wrong format of STROP matrix"
    lst = s.split()  # Split as a list of strings
    assert len(lst) > 2, err_msg
    nrows, ncolumns = int(lst[0]), int(lst[1])
    lst_float = [float(x) for x in lst[2:]]
    assert len(lst_float) == nrows * ncolumns, err_msg
    assert all(0 <= x <= 1.0 for x in lst_float), err_msg
    return np.reshape(np.array(lst_float), shape=(nrows, ncolumns))


def check_matrix(matrix: str, sol: set[str] = set()) -> bool:
    """Check that the matrix generates the set of solutions."""
    s = Polygon(str2OccMatrix(matrix))
    for tree in s.instances:
        str_tree = str(tree)
        if str_tree not in sol:
            return False
        sol.remove(str_tree)
    return len(sol) == 0


class TestSTROP(unittest.TestCase):
    def test_check_matrices_and_solutions(self):
        self.assertTrue(check_matrix(bool_matrix1, {sol1_1, sol1_2}))
        self.assertTrue(check_matrix(bool_matrix2))
        self.assertTrue(check_matrix(bool_matrix3, {sol3_1, sol3_2, sol3_3, sol3_4}))
        self.assertTrue(check_matrix(bool_matrix4))


if __name__ == "__main__":
    unittest.main()

bool_matrix1 = """
6 7
0 0 0 0 0 0 0
0 0 0 1 1 0 0 
0 0 1 1 1 1 0
0 1 1 1 1 1 1
0 0 0 1 1 1 1
0 0 0 1 0 1 0"""

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

bool_matrix2 = """
10 8
0 0 0 1 1 0 1 1
0 1 1 1 1 1 1 0
0 1 1 1 1 1 1 0
0 0 0 1 1 1 1 1
1 1 1 1 1 1 1 1
0 1 1 1 1 1 1 1
0 1 1 1 1 1 0 0
0 1 1 1 1 1 0 0
0 1 1 0 0 1 0 0
0 1 1 0 0 0 0 0"""

#########################################
bool_matrix3 = """
8 7
0 0 0 1 0 0 0
0 0 1 1 1 0 0
0 1 1 0.3 1 1 0
1 1 1 1 1 1 1
0 1 1 1 1 1 0
0 0 1 1 1 0 0
0 0 0 1 0 0 0
0 0 0 0 0 0 0
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
bool_matrix4 = """
3 3
1 0 1
0 1 1
1 1 1
"""
#######################################################
