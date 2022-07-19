import unittest

from frame.die.die import Die
from frame.geometry.geometry import Point, Shape


class MyTestCase(unittest.TestCase):

    def test_read_dies(self):
        d = Die("5.5x2")
        self.assertEqual(d.width, 5.5)
        self.assertEqual(d.height, 2.0)
        self.assertEqual(len(d.regions), 0)
        self.assertEqual(len(d.ground_regions), 1)
        r = d.ground_regions[0]
        self.assertEqual(r.center, Point(2.75, 1))
        self.assertEqual(r.shape, Shape(5.5, 2.0))

        d = Die(die1)
        self.assertEqual(d.width, 30)
        self.assertEqual(d.height, 50)
        self.assertEqual(len(d.regions), 0)
        self.assertEqual(len(d.ground_regions), 1)

        # Die with illegal height (-20)
        self.assertRaises(AssertionError, Die, die2, from_text=True)

        d = Die(die3)
        self.assertEqual(d.width, 30)
        self.assertEqual(d.height, 20)
        self.assertEqual(len(d.regions), 1)
        self.assertEqual(len(d.ground_regions), 3)

        # One of the rectangles is outside the die
        self.assertRaises(AssertionError, Die, die4, from_text=True)

        d = Die(die5)
        self.assertEqual(len(d.regions), 2)
        self.assertEqual(len(d.ground_regions), 4)

        # Contains a ground region
        self.assertRaises(AssertionError, Die, die6)


if __name__ == '__main__':
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
rectangles: [[5,8,10,12,'BRAM']]
"""

die4 = """
width: 30
height: 20
rectangles: [[5,8,10,12,'BRAM'], [28,10,10,10,'DSP']]
"""

die5 = """
width: 30
height: 20
rectangles: [[5,8,10,12,'BRAM'], [25,14,10,12,'DSP']]
"""

die6 = """
width: 30
height: 20
rectangles: [[5,8,10,12,'BRAM'], [25,14,10,12,'Ground']]
"""
