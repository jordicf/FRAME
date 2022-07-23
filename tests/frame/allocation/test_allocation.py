import unittest

from frame.allocation.allocation import Allocation


class MyTestCase(unittest.TestCase):

    def test_allocation(self):
        self.assertRaises(AssertionError, Allocation, alloc1)
        self.assertRaises(AssertionError, Allocation, alloc2)
        self.assertRaises(AssertionError, Allocation, alloc3)

        a = Allocation(alloc4)
        self.assertAlmostEqual(a.area("M1"), 27.4, places=4)
        self.assertAlmostEqual(a.center("M1").x, 3.83576, places=4)
        self.assertAlmostEqual(a.center("M1").y, 5.91970, places=4)

        self.assertAlmostEqual(a.area("M2"), 36.8, places=4)
        self.assertAlmostEqual(a.center("M2").x, 4.97826, places=4)
        self.assertAlmostEqual(a.center("M2").y, 4.73913, places=4)

        self.assertAlmostEqual(a.area("M3"), 10.8, places=4)
        self.assertAlmostEqual(a.center("M3").x, 1.5, places=4)
        self.assertAlmostEqual(a.center("M3").y, 3.0, places=4)

        self.assertAlmostEqual(a.area(["M1", "M2"]), 64.2, places=4)
        self.assertAlmostEqual(a.area(["M1", "M2", "M3"]), 75, places=4)

        m1m2c = a.center(["M1", "M2"])
        self.assertAlmostEqual(m1m2c.x, 4.49065, places=4)
        self.assertAlmostEqual(m1m2c.y, 5.24298, places=4)


if __name__ == '__main__':
    unittest.main()

# Allocation with overlap
alloc1 = """
[
  [[1.5,3,3,7], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""

# Allocation with occupancy > 1.0 in one rectangle
alloc2 = """
[
  [[1.5,3,3,6], {M1: 0.3, M3: 0.8}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""

# Allocation not in the positive quadrant
alloc3 = """
[
  [[1,3,3,6], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""

alloc4 = """
[
  [[1.5,3,3,6], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""
