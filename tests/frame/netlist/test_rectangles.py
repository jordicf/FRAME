# (c) Jordi Cortadella 2025
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
from pathlib import Path
from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Point, Shape, Rectangle


class TestRectangles(unittest.TestCase):
    netlist: Netlist  # The netlist for the test

    def setUp(self) -> None:
        file = str(Path(__file__).resolve().parents[0] / "netlist_rect.yml")
        self._netlist = Netlist(file)

    def test_rectangles(self):
        net = self._netlist
        self.assertEqual(net.num_rectangles, 8)
        
        # We visit all modules of the netlist
        module_names = {m.name for m in net.modules}
        set_names = {"A", "B", "C", "D", "E", "X", "Y", "Z"}
        self.assertEqual(module_names, set_names)

        # We visit all rectangles of all modules and calculate the total area
        area = 0
        for m in net.modules:
            for r in m.rectangles:
                area += r.area

        # Manual calculation of area
        area_rectangles = (15 + 6 + 4) + 16 + (18 + 4) + 25 + 8
        self.assertEqual(area, area_rectangles)

        # We redefine the rectangles of A
        r1 = Rectangle(center=Point(7, 8), shape=Shape(2, 2))
        r2 = Rectangle(ll=Point(4.5, 7.5), ur=Point(5.5, 10.5))
        net.assign_rectangles_module("A", [r1, r2])
        self.assertEqual(net.num_rectangles, 7)

        # We recalculate the area of the rectangles
        area = 0
        for m in net.modules:
            for r in m.rectangles:
                area += r.area

        area_rectangles += 4 + 3 - (15 + 6 + 4)
        self.assertEqual(area, area_rectangles)


if __name__ == "__main__":
    unittest.main()
