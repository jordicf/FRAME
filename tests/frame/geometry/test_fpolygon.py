# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
from pprint import pprint
from frame.geometry.fpolygon import FPolygon, XY_Box, StropDecomposition


class TestFPolygon(unittest.TestCase):
    def setUp(self) -> None:
        self.r1 = FPolygon([XY_Box(1, 5, 1, 4)])
        self.r2 = FPolygon([XY_Box(2, 4, 3, 6)])
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

    def test_strop(self) -> None:
        r1 = FPolygon(
            [
                XY_Box(3, 5, 1, 4.5),
                XY_Box(0, 3, 2, 4),
                XY_Box(1, 3, 4, 6),
                XY_Box(5, 7, 3, 6),
                XY_Box(3, 4, 0, 1),
                XY_Box(7, 8, 2, 5),
            ]
        )
        pprint(f"r1 = {r1}")
        self.assertEqual(r1.area, 27)
        strop = r1.largest_strop(FPolygon([XY_Box(3, 5, 1, 4.5)]))
        self.assertAlmostEqual(strop.similarity, 19.5 / 27, 7)
        pprint(f"strop = {strop}")
        self.assertEqual(strop.area(), 19.5)
        self.assertEqual(strop.num_branches, 4)

    def test_strop_reduction(self) -> None:
        r = FPolygon([XY_Box(0, 2, 0, 4), XY_Box(2, 4, 0, 1), XY_Box(2, 4, 3, 4)])
        self.assertEqual(r.area, 12)
        strop = r.calculate_best_strop()
        assert strop is not None
        self.assertEqual(r.area, strop.area())
        self.assertEqual(strop.num_branches, 2)
        pprint(f"r = {r}")
        pprint(f"strop = {strop}")
        new_strop = strop.reduce_branches("E")
        pprint(f"new_strop = {new_strop}")
        self.assertEqual(new_strop.area(), 12)
        self.assertEqual(new_strop.num_branches, 1)
        self.assertAlmostEqual(new_strop.similarity, 5/7, 7)
        new_strop = new_strop.reduce_branches("E")
        pprint(f"new_strop = {new_strop}")
        self.assertEqual(new_strop.area(), 12)
        self.assertEqual(new_strop.num_branches, 0)
        self.assertAlmostEqual(new_strop.similarity, 5/7, 7)
        
        r = FPolygon(
            [
                XY_Box(3, 9, 3, 10),
                XY_Box(6, 12, 6, 9),
                XY_Box(9, 13, 7, 9),
                XY_Box(3, 11, 3, 5),
                XY_Box(1, 3, 4, 5),
                XY_Box(3, 4, 1, 3),
                XY_Box(3, 5, 10, 11),
                XY_Box(6, 9, 10, 12),
                XY_Box(5, 6, 1, 7),
                XY_Box(8, 9, 1, 7),
                XY_Box(1, 4, 6, 9),
            ]
        )
        self.assertEqual(r.area, 79)
        strop = r.calculate_best_strop()
        assert strop is not None
        self.assertEqual(r.area, strop.area())
        nbranches = 10
        self.assertEqual(strop.num_branches, nbranches)
        pprint(f"r = {r}")
        pprint(f"strop = {strop}")
        new_strop = strop.dup()
        i = 1
        while new_strop.num_branches > 3:
            new_strop = new_strop.reduce()
            nbranches -= 1
            pprint(f"new_strop {i} = {new_strop}")
            pprint(f"similarity = {new_strop.similarity}")
            self.assertAlmostEqual(new_strop.area(), 79, 7)
            self.assertEqual(new_strop.num_branches, nbranches)
            i += 1
        self.assertAlmostEqual(new_strop.similarity, 71.25/86.75, 7)
        
        while new_strop.num_branches > 0:
            new_strop = new_strop.reduce()
            nbranches -= 1
            pprint(f"new_strop {i} = {new_strop}")
            pprint(f"similarity = {new_strop.similarity}")
            self.assertAlmostEqual(new_strop.area(), 79, 7)
            self.assertEqual(new_strop.num_branches, nbranches)
            i += 1
            
        self.assertAlmostEqual(new_strop.similarity, 0.6538, 3)
            


if __name__ == "__main__":
    unittest.main()
