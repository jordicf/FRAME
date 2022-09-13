import unittest

from frame.netlist.netlist import Netlist


class TestNetlist(unittest.TestCase):
    def setUp(self) -> None:
        netlist_yaml = """
        Modules: {
          B1: {
            area: 6,
            center: [2,3]
          },
          B2: {
            area: 3,
            center: [3,3],
          },
          B3: {
            area: 5,
            center: [1,1]
          },
          B4: {
            area: 2,
            rectangles: [[3,0.5,2,1]],
          },
          B5: {
            rectangles: [[3,0.5,2,1]],
            hard: true,
            fixed: false
          }
        }

        Nets: [
          [B1, B2, 5],
          [B2, B3, B4, 10],
          [B4, B1, B2]
        ]
        """
        self.netlist = Netlist(netlist_yaml)
        self.netlist.get_module("B4").calculate_center_from_rectangles()

    def test_num_modules(self):
        self.assertEqual(self.netlist.num_modules, 5)

    def test_num_edges(self):
        self.assertEqual(self.netlist.num_edges, 3)

    def test_wire_length(self):
        self.assertAlmostEqual(self.netlist.wire_length, 5 + 42.6733 + 3.66439, places=4)


if __name__ == '__main__':
    unittest.main()
