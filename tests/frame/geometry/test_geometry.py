# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest

from frame.geometry.geometry import Point, Shape, Rectangle, BoundingBox, create_strop
from frame.utils.keywords import KW


class TestPoint(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Point(1, 2)
        self.q = Point(-1.5, 0)
        self.r = Point()

    def test_eq(self) -> None:
        self.assertEqual(self.p, Point(1, 2))
        self.assertEqual(self.q, Point(-1.5, 0))
        self.assertEqual(self.r, Point(0, 0))
        self.assertNotEqual(self.p, self.q)

    def test_neg(self) -> None:
        self.assertEqual(-self.p, Point(-1, -2))
        self.assertEqual(-self.q, Point(1.5, 0))

    def test_add(self) -> None:
        self.assertEqual(self.p + 0, Point(1, 2))
        self.assertEqual(self.q + 1.5, Point(0, 1.5))
        self.assertEqual(self.q + 1.5, 1.5 + self.q)
        self.assertEqual(self.p + self.q, Point(-0.5, 2))
        self.assertEqual(self.p + self.q, self.q + self.p)
        self.assertEqual(self.r + self.r, self.r)

    def test_sub(self) -> None:
        self.assertEqual(self.p - 0, Point(1, 2))
        self.assertEqual(self.q - 1.5, Point(-3, -1.5))
        self.assertEqual(1.5 - self.q, Point(3, 1.5))
        self.assertEqual(self.p - self.q, Point(2.5, 2))
        self.assertEqual(self.q - self.p, Point(-2.5, -2))
        self.assertEqual(self.p - self.p, self.q - self.q)
        self.assertEqual(self.p - self.p, self.r)
        self.assertEqual(self.r - self.r, self.r)
        self.assertNotEqual(self.p - self.q, self.q - self.p)

    def test_mul(self) -> None:
        self.assertEqual(self.p * 2, Point(2, 4))
        self.assertEqual(self.p * 0.5, Point(0.5, 1))
        self.assertEqual(self.p * 1, self.p)
        self.assertEqual(self.p * -1, -self.p)
        self.assertEqual(2 * self.q, Point(-3, 0))
        self.assertEqual(0.5 * self.q, Point(-0.75, 0))
        self.assertEqual(1 * self.q, self.q)
        self.assertEqual(-1 * self.q, -self.q)
        self.assertEqual(self.p * self.r, self.r)
        self.assertEqual(self.p * self.q, self.q * self.p)

    def test_pow(self) -> None:
        self.assertEqual(self.p**2, Point(1, 4))
        self.assertEqual(self.q**2, Point(2.25, 0))
        self.assertEqual(self.p**0, self.r + 1)

    def test_div(self) -> None:
        self.assertEqual(self.p / 2, Point(0.5, 1))
        self.assertEqual(self.p / 1, self.p)
        self.assertEqual(self.p / -1, -self.p)
        self.assertEqual(self.q / -1.5, Point(1, 0))
        self.assertEqual(self.r / 10, self.r)
        self.assertEqual(2 / self.p, Point(2, 1))
        self.assertEqual(1 / self.p, Point(1, 0.5))
        self.assertEqual(-1 / self.p, Point(-1, -0.5))

    def test_dot_product(self) -> None:
        self.assertEqual(self.p & self.q, -1.5)
        self.assertEqual(self.p & self.q, self.q & self.p)
        self.assertEqual(self.r & self.r, 0)
        self.assertEqual(self.p & Point(-2, 1), 0)

    def test_norm(self) -> None:
        self.assertAlmostEqual(self.p.norm(), 2.2360679775)
        self.assertEqual(self.q.norm(), 1.5)
        self.assertEqual(self.r.norm(), 0)

    def test_str(self) -> None:
        self.assertEqual(str(self.p), "Point(x=1, y=2)")
        self.assertEqual(str(self.q), "Point(x=-1.5, y=0)")
        self.assertEqual(str(self.r), "Point(x=0, y=0)")
        self.assertEqual(repr(self.p), "Point(x=1, y=2)")
        self.assertEqual(repr(self.q), "Point(x=-1.5, y=0)")
        self.assertEqual(repr(self.r), "Point(x=0, y=0)")

    def test_iter(self) -> None:
        self.assertEqual(tuple(self.p), (1, 2))
        qx, qy = self.q
        self.assertEqual(qx, -1.5)
        self.assertEqual(qy, 0)
        self.assertEqual(list(self.r), [0, 0])


class TestRectangle(unittest.TestCase):
    def test_bad_rectangles(self):
        # Bad constructors
        self.assertRaises(Exception, Rectangle, unknown=Point(1, 1))
        self.assertRaises(AssertionError, Rectangle, center=Shape(1, 1))
        self.assertRaises(AssertionError, Rectangle, shape=Point(1, 1))

    def test_area_rectangles(self):
        r = Rectangle(center=Point(5, 3), shape=Shape(4, 2))
        self.assertEqual(r.area, 8)
        self.assertTrue(r.point_inside(Point(3.1, 2.5)))
        self.assertFalse(r.point_inside(Point(2.9, 2.1)))
        self.assertFalse(r.point_inside(Point(3.1, 1.9)))
        self.assertEqual(r.bounding_box, BoundingBox(ll=Point(3, 2), ur=Point(7, 4)))
        r.shape = Shape(3, 1)
        self.assertEqual(r.area, 3)
        self.assertEqual(
            r.bounding_box, BoundingBox(ll=Point(3.5, 2.5), ur=Point(6.5, 3.5))
        )
        r.center = Point(30, 40)
        self.assertEqual(r.area, 3)
        self.assertEqual(
            r.bounding_box, BoundingBox(ll=Point(28.5, 39.5), ur=Point(31.5, 40.5))
        )

    def test_overlap_rectangles(self):
        Rectangle.set_epsilon(10e-12)
        r1 = Rectangle(center=Point(2, 3), shape=Shape(2, 4))
        r2 = Rectangle(center=Point(5, 5), shape=Shape(4, 2))
        r3 = Rectangle(center=Point(8, 5), shape=Shape(4, 6))
        r4 = Rectangle(center=Point(6, 2), shape=Shape(4, 2))
        self.assertFalse(r1.overlap(r2))
        self.assertFalse(r1.overlap(r3))
        self.assertFalse(r1.overlap(r4))
        self.assertTrue(r2.overlap(r3))
        self.assertFalse(r2.overlap(r4))
        self.assertTrue(r3.overlap(r4))
        self.assertEqual(r1.area_overlap(r2), 0)
        self.assertEqual(r1.area_overlap(r3), 0)
        self.assertEqual(r1.area_overlap(r4), 0)
        self.assertEqual(r2.area_overlap(r3), 2)
        self.assertEqual(r2.area_overlap(r4), 0)
        self.assertEqual(r3.area_overlap(r4), 2)

        # Checking __mul__
        self.assertIsNone(r1 * r2)
        self.assertIsNone(r1 * r3)
        self.assertIsNone(r1 * r4)
        self.assertIsNone(r2 * r4)
        r5 = r2 * r3
        r6 = r3 * r4
        r5_inter = Rectangle(**{KW.CENTER: Point(6.5, 5.0), KW.SHAPE: Shape(1, 2)})
        r6_inter = Rectangle(**{KW.CENTER: Point(7, 2.5), KW.SHAPE: Shape(2, 1)})
        self.assertTrue(r5 == r5_inter)
        self.assertTrue(r6 == r6_inter)

    def test_inside_rectangles(self):
        r1 = Rectangle(center=Point(2, 3), shape=Shape(4, 6))
        r2 = Rectangle(center=Point(3, 5), shape=Shape(2, 2))
        r3 = Rectangle(center=Point(4, 6), shape=Shape(2, 2))
        self.assertTrue(r1.point_inside(Point(3, 4)))
        self.assertTrue(r1.point_inside(Point(4, 6)))
        self.assertTrue(r1.point_inside(Point(0, 0)))
        self.assertFalse(r1.point_inside(Point(5, 3)))
        self.assertFalse(r1.point_inside(Point(2, 7)))
        self.assertTrue(r2.is_inside(r1))
        self.assertFalse(r3.is_inside(r1))

    def test_trunked_polygons(self):
        Rectangle.set_epsilon(10e-12)
        r1 = Rectangle(center=Point(2, 3.5), shape=Shape(2, 3))
        r2 = Rectangle(center=Point(4.5, 4.5), shape=Shape(3, 1))
        r3 = Rectangle(center=Point(6, 9.5), shape=Shape(2, 3))
        r4 = Rectangle(center=Point(7, 12), shape=Shape(2, 2))
        r5 = Rectangle(center=Point(9, 12), shape=Shape(2, 6))
        r6 = Rectangle(center=Point(9.5, 15.5), shape=Shape(1, 1))
        r7 = Rectangle(center=Point(11, 12.5), shape=Shape(2, 3))
        r8 = Rectangle(center=Point(10.5, 9.5), shape=Shape(1, 1))
        r9 = Rectangle(center=Point(9, 8), shape=Shape(1, 2))

        self.assertTrue(create_strop([r1]))
        self.assertTrue(create_strop([r5]))
        self.assertTrue(create_strop([r2, r1]))
        self.assertFalse(create_strop([r2, r3]))
        self.assertFalse(create_strop([r3, r4]))
        self.assertTrue(create_strop([r4, r5]))
        polygon = [r7, r8, r9, r5, r6, r4]
        self.assertTrue(create_strop(polygon))
        self.assertEqual(polygon[0], r5)
        self.assertEqual(r5.location, Rectangle.StropLocation.TRUNK)
        self.assertEqual(r6.location, Rectangle.StropLocation.NORTH)
        self.assertEqual(r7.location, Rectangle.StropLocation.EAST)
        self.assertEqual(r8.location, Rectangle.StropLocation.EAST)
        self.assertEqual(r9.location, Rectangle.StropLocation.SOUTH)
        self.assertEqual(r4.location, Rectangle.StropLocation.WEST)


if __name__ == "__main__":
    unittest.main()
