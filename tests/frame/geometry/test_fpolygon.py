# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
from frame.geometry.fpolygon import FPolygon


class TestFPolygon(unittest.TestCase):
    def setUp(self) -> None:
        self.r1 = FPolygon([(1, 5, 1, 4)])
        self.r2 = FPolygon([(2, 4, 3, 6)])
        self.r3 = self.r1 | self.r2
        self.r4 = self.r1 & self.r2
        self.r5 = self.r1 - self.r2
        self.empty = FPolygon()

    def test_area(self) -> None:
        self.assertEqual(self.r1.area, 12)
        self.assertEqual(self.r2.area, 6)
        self.assertEqual(self.r3.area, 16)
        self.assertEqual(self.r4.area, 2)
        self.assertEqual(self.r5.area, 10)
        self.assertEqual(self.empty.area, 0)

    def test_num_rectangles(self) -> None:
        self.assertEqual(self.r1.num_rectangles, 1)
        self.assertEqual(self.r2.num_rectangles, 1)
        self.assertEqual(self.r3.num_rectangles, 2)
        self.assertEqual(self.r4.num_rectangles, 1)
        self.assertEqual(self.r5.num_rectangles, 3)
        self.assertEqual(self.empty.num_rectangles, 0)

    def test_copy(self) -> None:
        c = self.r5.copy()
        self.assertEqual(c.area, 10)
        self.assertEqual(c.num_rectangles, 3)
        
    def test_strop(self) -> None:
        r1 = FPolygon([(3,5,1,4.5), (0,3,2,4), (1,3,4,6), (5,7,3,6), (3,4,0,1), (7,8,2,5)])
        print(f"r1 = {r1}")
        self.assertEqual(r1.num_rectangles, 9)
        self.assertEqual(r1.area, 27)
        strop = r1.largest_strop(FPolygon([(3,5,1,4.5)]))
        print(f"strop = {strop}")
        self.assertEqual(strop.area, 19.5)
        self.assertEqual(strop.num_rectangles, 5)
        self.assertEqual(r1.jaccard_similarity(strop), 19.5/27)


if __name__ == "__main__":
    unittest.main()
