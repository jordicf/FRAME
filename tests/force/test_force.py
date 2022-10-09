# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
import math

from frame.geometry.geometry import Point

from tools.force.fruchterman_reingold import circle_circle_intersection_area


class TestForce(unittest.TestCase):
    def test_circle_circle_intersection_area(self):
        self.assertAlmostEqual(circle_circle_intersection_area(Point(0, 0), 1, Point(0, 0), 1), math.pi)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(0, 0), 1, Point(0, 0), 2), math.pi)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(3, 3), 3, Point(2, 2), 1), math.pi)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(2, 4), 2, Point(3, 9), 3), 0)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(0, 0), 2, Point(-1, 1), 2), 7.0297, 4)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(1, 3), 2, Point(-1, 3), 2.19), 5.7247, 4)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(1, 3), 2, Point(1, 3), 2.19), 12.5664, 4)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(0, 0), 2, Point(-1, 0), 2), 8.6084, 4)
        self.assertAlmostEqual(circle_circle_intersection_area(Point(4, 3), 2, Point(2.5, 3.5), 1.4), 3.7536, 4)


if __name__ == '__main__':
    unittest.main()
