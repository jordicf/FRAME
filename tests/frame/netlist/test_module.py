# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest

from frame.geometry.geometry import Point, Shape, Rectangle
from frame.netlist.module import Module


class TestModule(unittest.TestCase):

    def test_bad_module(self):
        # Bad constructors
        self.assertRaises(AssertionError, Module, "my_module", unknown_keyword=3)
        self.assertRaises(AssertionError, Module, "3_bad_name_4")
        self.assertRaises(AssertionError, Module, "good_name", area="no_number")

    def test_module_rectangles(self):
        b = Module("my_module", area={'dsp': 10, 'lut': 6, 'bram': 20})
        self.assertEqual(b.num_rectangles, 0)

        # Create 3 rectangles
        b.add_rectangle(Rectangle(center=Point(4, 5), shape=Shape(4, 6)))
        b.add_rectangle(Rectangle(center=Point(8, 3), shape=Shape(4, 2)))
        b.add_rectangle(Rectangle(center=Point(8, 8), shape=Shape(2, 2)))
        self.assertEqual(b.num_rectangles, 3)
        self.assertEqual(b.area_rectangles, 36)
        b.calculate_center_from_rectangles()
        self.assertTrue(5.3333 < b.center.x < 5.334)
        self.assertTrue(4.8888 < b.center.y < 4.8889)

        # Create only one square from the total area\
        b.create_square()
        self.assertEqual(b.num_rectangles, 1)
        self.assertTrue(35.999999 < b.area_rectangles < 36.000001)

    def test_module_regions(self):
        b = Module("my_module", area=100)
        self.assertEqual(b.area(), 100)
        self.assertEqual(b.area('_'), 100)
        self.assertEqual(b.area('wrong_region'), 0)
        b = Module("my_module", area={'dsp': 10, 'lut': 6, 'bram': 20})
        self.assertEqual(b.area(), 36)
        self.assertEqual(b.area('wrong_region'), 0)
        self.assertEqual(b.area('dsp'), 10)
        self.assertEqual(b.area('lut'), 6)
        self.assertEqual(b.area('bram'), 20)

    def test_soft_hard_fixed(self):
        b = Module("my_module", hard=True)

        self.assertFalse(b.is_soft)
        self.assertTrue(b.is_hard)
        self.assertFalse(b.is_fixed)
        for rectangle in b.rectangles:
            self.assertFalse(rectangle.fixed)

        b.is_fixed = True
        self.assertFalse(b.is_soft)
        self.assertTrue(b.is_hard)
        self.assertTrue(b.is_fixed)
        for rectangle in b.rectangles:
            self.assertTrue(rectangle.fixed)

        b.is_fixed = False
        self.assertFalse(b.is_soft)
        self.assertTrue(b.is_hard)
        self.assertFalse(b.is_fixed)
        for rectangle in b.rectangles:
            self.assertFalse(rectangle.fixed)


if __name__ == '__main__':
    unittest.main()
