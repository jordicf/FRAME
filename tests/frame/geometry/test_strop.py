"""
Testing the stop module.
A set of matrices matrix? are given, with their corresponding STrOP instances (sol?_*). The test checks that all these solutions are generated.
"""
import strop

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


def check_matrix(matrix: str, sol: set[str] = set()) -> None:
    s = strop.Strop(matrix)
    for tree in s.instances():
        s = str(tree)
        assert s in sol
        sol.remove(s)
    assert len(sol) == 0


def test_strop() -> None:
    check_matrix(matrix1, {sol1_1, sol1_2})
    check_matrix(matrix2)
    check_matrix(matrix3, {sol3_1, sol3_2, sol3_3, sol3_4})
    check_matrix(matrix4)
