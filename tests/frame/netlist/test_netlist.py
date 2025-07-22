# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
from pathlib import Path
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point, Shape, Rectangle


class TestNetlist(unittest.TestCase):
    def setUp(self) -> None:
        file = str(Path(__file__).resolve().parents[0] / "netlist_basic.yml")
        self.netlist = Netlist(file)
        self.netlist.get_module("B4").calculate_center_from_rectangles()

    def test_num_modules(self):
        self.assertEqual(self.netlist.num_modules, 5)

    def test_num_edges(self):
        self.assertEqual(self.netlist.num_edges, 3)

    def test_wire_length(self):
        self.assertAlmostEqual(
            self.netlist.wire_length, 5 + 42.6733 + 3.66439, places=4
        )


class TestRectangles(unittest.TestCase):
    def setUp(self) -> None:
        file = str(Path(__file__).resolve().parents[0] / "netlist_rect.yml")
        self.netlist = Netlist(file)

    def test_num_rectangles(self):
        net = self.netlist
        self.assertEqual(net.num_rectangles, 8)
        # We redefine the rectangles of A
        r1 = Rectangle(center=Point(7,8), shape=Shape(2,2))
        r2 = Rectangle(ll=Point(4.5, 7.5), ur=Point(5.5, 10.5))
        net.assign_rectangles_module('A', [r1, r2])
        self.assertEqual(net.num_rectangles, 7)


if __name__ == "__main__":
    unittest.main()
