# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape
from frame.netlist.netlist import Netlist


class TestDie(unittest.TestCase):
    def test_read_dies(self):
        d = Die("5.5x2")
        self.assertEqual(d.width, 5.5)
        self.assertEqual(d.height, 2.0)
        self.assertEqual(len(d.specialized_regions), 0)
        self.assertEqual(len(d.ground_regions), 1)
        r = d.ground_regions[0]
        self.assertEqual(r.center, Point(2.75, 1))
        self.assertEqual(r.shape, Shape(5.5, 2.0))

        d = Die(die1)
        self.assertEqual(d.width, 30)
        self.assertEqual(d.height, 50)
        self.assertEqual(len(d.specialized_regions), 0)
        self.assertEqual(len(d.ground_regions), 1)

        # Die with illegal height (-20)
        self.assertRaises(AssertionError, Die, die2)

        d = Die(die3)
        self.assertEqual(d.width, 30)
        self.assertEqual(d.height, 20)
        self.assertEqual(len(d.specialized_regions), 1)
        self.assertEqual(len(d.ground_regions), 3)

        # One of the rectangles is outside the die
        self.assertRaises(AssertionError, Die, die4)

        d = Die(die5)
        self.assertEqual(len(d.specialized_regions), 2)
        self.assertEqual(len(d.ground_regions), 4)

        # Contains a ground region
        self.assertRaises(AssertionError, Die, die6)

    def test_die_rectangles(self):
        n = Netlist(netlist_die7)
        d = Die(die7, n)
        self.assertEqual(len(d.specialized_regions), 2)
        self.assertEqual(len(d.blockages), 1)
        self.assertEqual(len(d.fixed_regions), 3)
        self.assertEqual(len(d.ground_regions), 13)
        d.split_refinable_regions(3, 23)
        self.assertEqual(len(d.specialized_regions), 3)
        self.assertEqual(len(d.blockages), 1)
        self.assertEqual(len(d.fixed_regions), 3)
        self.assertEqual(len(d.ground_regions), 20)

        d = Die(die1)
        self.assertEqual(len(d.specialized_regions), 0)
        self.assertEqual(len(d.blockages), 0)
        self.assertEqual(len(d.fixed_regions), 0)
        self.assertEqual(len(d.ground_regions), 1)
        d.split_refinable_regions(2, 8)
        self.assertEqual(len(d.specialized_regions), 0)
        self.assertEqual(len(d.blockages), 0)
        self.assertEqual(len(d.fixed_regions), 0)
        self.assertEqual(len(d.ground_regions), 8)

    def test_die_io_segments(self):
        n = Netlist(netlist_die8)
        d = Die(die8, n)


if __name__ == "__main__":
    unittest.main()

die1 = """
width: 30
height: 50
"""

die2 = """
width: 30
height: -20
"""

die3 = """
width: 30
height: 20
regions: [[5,8,10,12,'BRAM']]
"""

die4 = """
width: 30
height: 20
regions: [[5,8,10,12,'BRAM'], [28,10,10,10,'DSP']]
"""

die5 = """
width: 30
height: 20
regions: [[5,8,10,12,'BRAM'], [25,14,10,12,'DSP']]
"""

die6 = """
width: 30
height: 20
regions: [[5,8,10,12,'BRAM'], [25,14,10,12,'_']]
"""

die7 = """
width: 10
height: 9
regions: [[6, 7.5, 4, 1, "reg1"], [7, 1.5, 2, 3, "reg2"], [3, 3.5, 2, 3, "#"]]
"""

netlist_die7 = """
Modules : {
  M1: {
    fixed: true,
    rectangles: [[2,7,2,2]]
  },
  M2: {
    fixed: true,
    rectangles: [[8,5.5,2,1]]
  },
  M3: {
    area: 10
  },
  
  P1: {
    io_pin: true,
    length: 64
  },
  
  P2: {
    io_pin: true,
    fixed: true,
    rectangles: [5, 8, 10, 0],
  }
}
Nets: []
"""

die8 = """
width: 10
height: 9
io_segments: {
  P1: [[10, 5, 0, 2], 
       [6, 9, 3, 0]],
  P2: [0, 8, 10, 0]
}
"""

netlist_die8 = """
Modules : {
  M1: {
    fixed: true,
    rectangles: [[2,7,2,2]]
  },
  M2: {
    fixed: true,
    rectangles: [[8,5.5,2,1]]
  },
  M3: {
    area: 10
  },
  
  P1: {
    io_pin: true,
    length: 100
  },
  
  P2: {
    io_pin: true,
    fixed: true,
    rectangles: [[5, 8, 10, 0]],
  }
}
Nets: []
"""
