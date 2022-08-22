import unittest

from frame.allocation.allocation import Allocation
from frame.netlist.netlist import Netlist
from frame.utils.utils import read_yaml


class TestAllocation(unittest.TestCase):

    def test_allocation_errors(self):
        self.assertRaises(AssertionError, Allocation, alloc1)
        self.assertRaises(AssertionError, Allocation, alloc2)
        self.assertRaises(AssertionError, Allocation, alloc3)
        self.assertRaises(AssertionError, Allocation, alloc4)
        self.assertRaises(AssertionError, Allocation, alloc5)
        self.assertRaises(AssertionError, Allocation, alloc6)

    def test_allocation_methods(self):
        a = Allocation(alloc7)
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

    def test_module2rectangle(self):
        n = Netlist(netlist1)
        for m, nr in [("M1", 1), ("M2", 1), ("M3", 0)]:
            self.assertEqual(n.get_module(m).num_rectangles, nr)
        m2r = read_yaml(mod2rect)
        n.assign_rectangles(m2r)
        for m, nr in [("M1", 2), ("M2", 1), ("M3", 2)]:
            self.assertEqual(n.get_module(m).num_rectangles, nr)
        n = Netlist(netlist2)
        for m, nr in [("M1", 1), ("M2", 1)]:
            self.assertEqual(n.get_module(m).num_rectangles, nr)
        self.assertRaises(AssertionError, n.assign_rectangles, m2r)

    def test_refinement(self):
        a = Allocation(alloc7)
        self.assertEqual(3, a.num_rectangles)
        a2 = a.refine(0.7, 1)
        self.assertEqual(5, a2.num_rectangles)
        a4 = a.refine(0.7, 2)
        self.assertEqual(9, a4.num_rectangles)

    def test_must_be_refined(self):
        a = Allocation(alloc8)
        self.assertTrue(a.must_be_refined(0.95))
        self.assertTrue(a.must_be_refined(0.85))
        self.assertTrue(a.must_be_refined(0.75))
        self.assertFalse(a.must_be_refined(0.65))

    def test_compatible_netlist(self):
        a = Allocation(alloc7)
        n1 = Netlist(netlist1)
        n2 = Netlist(netlist2)
        self.assertTrue(a.check_compatible(n1))
        self.assertFalse(a.check_compatible(n2))

    def test_uniform_refinement(self):
        a = Allocation(alloc7)
        self.assertEqual(3, a.num_rectangles)
        a4 = a.refine(0.7, 2)
        self.assertEqual(9, a4.num_rectangles)
        a_uniform = a4.uniform_refinement_depth()
        self.assertEqual(12, a_uniform.num_rectangles)

    def test_griddify(self):
        a = Allocation(alloc_griddify)
        self.assertEqual(12, a.num_rectangles)
        new_a = a.griddify()
        self.assertEqual(42, new_a.num_rectangles)
        # make the last rectangle fixed
        a.allocations[-1].rect.fixed = True
        new_a = a.griddify()
        self.assertEqual(40, new_a.num_rectangles)



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

# Allocation with occupancy < 0 in one rectangle
alloc2 = """
[
  [[1.5,3,3,6], {M1: -0.1, M3: 0.8}],
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
  [[1.5,3,3,6,8], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""

alloc5 = """
[
  [[1.5,3,3,6], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,'8',4], {M1: 0.5, M2: 0.4}]
]
"""

alloc6 = """
[
  [[1.5,3,3,6], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: '0.8'}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""

alloc7 = """
[
  [[1.5,3,3,6], {M1: 0.3, M3: 0.6}],
  [[5.5,3,5,6], {M1: 0.2, M2: 0.8}],
  [[4,8,8,4], {M1: 0.5, M2: 0.4}]
]
"""

alloc8 = """
[
  [[1.5,3,3,6], {M1: 0.2, M3: 0.8}],
  [[5.5,3,5,6], {M1: 0.1, M2: 0.9}],
  [[4,8,8,4], {M1: 0.3, M2: 0.7}]
]
"""
alloc_griddify = """
[
  [[1, 1.5, 2, 3], {M1: 1}],
  [[5.5, 1, 7, 2], {M2: 1}],
  [[10.5, 1, 3, 2], {M3: 1}],
  [[1, 4.5, 2, 3], {M4: 1}],
  [[7.5, 3.5, 5, 3], {M5: 1}],
  [[11, 3.5, 2, 3], {M6: 1}],
  [[2.5, 7, 5, 2], {M7: 1}],
  [[6.5, 7.5, 3, 5], {M8: 1}],
  [[10, 7.5, 4, 5], {M9: 1}],
  [[2.5, 9, 5, 2], {M10: 1}],
  [[6, 11, 12, 2], {M11: 1}],
  [[3.5, 4, 3, 4], {FIX: 1}]
]
"""

netlist1 = """
Modules: {
  M1: {
    area: 18,
    rectangles: [[3,3,6,3]],
    fixed: false
  },
  M2: {
    rectangles: [[4,2.5,4,5]],
    fixed: true
  },
  M3: {
    area: {"DSP": 20, "BRAM": 50, "LUT": 40}
  }
}

Nets: [
  [M1, M2, 5]
]
"""

netlist2 = """
Modules: {
  M1: {
    area: 18,
    rectangles: [[3,3,6,3]],
    fixed: false
  },
  M2: {
    rectangles: [[4,2.5,4,5]],
    fixed: true
  }
}

Nets: [
  [M1, M2, 5]
]
"""

mod2rect = """
{
M1: [[5, 6.5, 2, 4, "DSP"], [7, 4, 8, 10]],
M3: [[15, 8, 12, 6, "DSP"], [5, 16, 4.5, 8.4]]
}
"""

alloc_griddify = """
[
  [[1, 1.5, 2, 3], {M1: 1}],
  [[5.5, 1, 7, 2], {M2: 1}],
  [[10.5, 1, 3, 2], {M3: 1}],
  [[1, 4.5, 2, 3], {M4: 1}],
  [[7.5, 3.5, 5, 3], {M5: 1}],
  [[11, 3.5, 2, 3], {M6: 1}],
  [[2.5, 7, 5, 2], {M7: 1}],
  [[6.5, 7.5, 3, 5], {M8: 1}],
  [[10, 7.5, 4, 5], {M9: 1}],
  [[2.5, 9, 5, 2], {M10: 1}],
  [[6, 11, 12, 2], {M11: 1}],
  [[3.5, 4, 3, 4], {FIX: 1}]
]
"""