# (c) Antoni Pech-Alberich 2025
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
import numpy as np
from frame.netlist.netlist import Netlist
from tools.floorset_parser.floor_set_manager.manager import FloorSetInstance
from frame.geometry.geometry import Point
import tools.floorset_parser.floor_set_manager.utils.utils as fun


class TestFloorSetInstance(unittest.TestCase):
    def setUp(self):
        self.data = {
            'area_blocks': np.array([
                1024.,  512.,  256.,  768.,  768.,  256.,  768.,  512.,  768.,
                3072., 1536., 1024., 2560.,  512., 2304., 1024.,  512., 1920.,
                1152., 2560.,  512.
            ]),
            'b2b_connectivity': np.array([
                [0.000e+00, 1.000e+00, 128],
                [0.000e+00, 1.600e+01, 512],
                [1.000e+00, 1.400e+01, 1024],
                [2.000e+00, 4.000e+00, 1024],
                [2.000e+00, 1.500e+01, 512],
                [4.000e+00, 6.000e+00, 512],
                [8.000e+00, 1.100e+01, 128],
                [8.000e+00, 1.700e+01, 512],
                [1.100e+01, 1.900e+01, 512],
                [1.300e+01, 1.700e+01, 256],
                [1.400e+01, 1.700e+01, 1024],
                [1.400e+01, 1.800e+01, 128],
                [1.700e+01, 2.000e+01, 256]
            ]),
            'p2b_connectivity': np.array([
                [1.000e+00, 7.000e+00, 512],
                [1.000e+00, 1.500e+01, 512],
                [2.000e+00, 5.000e+00, 512],
                [3.000e+00, 0.000e+00, 512],
                [1.000e+01, 1.300e+01, 512],
                [1.000e+01, 1.400e+01, 512],
                [1.100e+01, 4.000e+00, 512],
                [1.100e+01, 1.700e+01, 512],
                [1.200e+01, 3.000e+00, 512],
                [1.200e+01, 1.400e+01, 512]
            ]),
            'pins_pos': np.array([
                [  0.,   0.],
                [ 33.,   0.],
                [ 88.,   0.],
                [132.,   0.],
                [143.,   0.],
                [  0., 160.],
                [ 22., 160.],
                [ 88., 160.],
                [110., 160.],
                [132., 160.],
                [143., 160.],
                [  0.,  11.],
                [  0.,  55.],
                [  0.,  77.],
                [  0., 121.],
                [  0., 132.],
                [  0., 154.],
                [152.,  11.],
                [152.,  22.],
                [152.,  88.],
                [152., 154.]
            ]),
            'placement_constraints': np.array([
                [ 0.,  0.,  0.,  3.,  0.],
                [ 0.,  0.,  1.,  0.,  8.],
                [ 0.,  0.,  2.,  2.,  0.],
                [ 0.,  0.,  3.,  1.,  0.],
                [ 0.,  0.,  3.,  0.,  0.],
                [ 0.,  0.,  2.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  1.,  1.,  0.],
                [ 0.,  0.,  0.,  0.,  0.],
                [ 1.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  6.],
                [ 0.,  0.,  0.,  2.,  0.],
                [ 0.,  0.,  0.,  0.,  5.],
                [ 0.,  1.,  1.,  2.,  0.],
                [ 0.,  0.,  0.,  0., 10.],
                [ 0.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  1.,  0.],
                [ 0.,  0.,  0.,  0.,  2.],
                [ 0.,  0.,  0.,  0.,  2.],
                [ 0.,  0.,  0.,  3.,  1.],
                [ 0.,  0.,  1.,  3.,  0.]
            ]),
            'vertex_blocks': np.array([
                [[  0.,   0.],
                    [  0.,  32.],
                    [ 32.,  32.],
                    [ 32.,   0.],
                    [  0.,   0.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 32.,   0.],
                    [ 32.,  32.],
                    [ 48.,  32.],
                    [ 48.,   0.],
                    [ 32.,   0.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 48.,  64.],
                    [ 48.,  80.],
                    [ 64.,  80.],
                    [ 64.,  64.],
                    [ 48.,  64.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 80.,  64.],
                    [ 80.,  80.],
                    [128.,  80.],
                    [128.,  64.],
                    [ 80.,  64.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 80.,  80.],
                    [ 80.,  96.],
                    [128.,  96.],
                    [128.,  80.],
                    [ 80.,  80.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 48.,  96.],
                    [ 48., 112.],
                    [ 64., 112.],
                    [ 64.,  96.],
                    [ 48.,  96.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 48., 112.],
                    [ 48., 160.],
                    [ 64., 160.],
                    [ 64., 112.],
                    [ 48., 112.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 64.,  80.],
                    [ 80.,  80.],
                    [ 80.,  48.],
                    [ 64.,  48.],
                    [ 64.,  80.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 64.,  48.],
                    [ 80.,  48.],
                    [ 80.,   0.],
                    [ 64.,   0.],
                    [ 64.,  48.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 80., 160.],
                    [128., 160.],
                    [128.,  96.],
                    [ 80.,  96.],
                    [ 80., 160.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[128., 160.],
                    [152., 160.],
                    [152.,  96.],
                    [128.,  96.],
                    [128., 160.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 64.,  64.],
                    [ 64.,   0.],
                    [ 48.,   0.],
                    [ 48.,  64.],
                    [ 64.,  64.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 48., 160.],
                    [ 48.,  96.],
                    [ 32.,  96.],
                    [ 32., 112.],
                    [  0., 112.],
                    [  0., 160.],
                    [ 48., 160.]],

                [[ 32.,  64.],
                    [ 48.,  64.],
                    [ 48.,  32.],
                    [ 32.,  32.],
                    [ 32.,  64.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[152.,  32.],
                    [152.,   0.],
                    [ 80.,   0.],
                    [ 80.,  32.],
                    [152.,  32.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 64., 160.],
                    [ 80., 160.],
                    [ 80.,  96.],
                    [ 64.,  96.],
                    [ 64., 160.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 80.,  96.],
                    [ 80.,  80.],
                    [ 48.,  80.],
                    [ 48.,  96.],
                    [ 80.,  96.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[152.,  48.],
                    [ 80.,  48.],
                    [ 80.,  64.],
                    [128.,  64.],
                    [128.,  96.],
                    [152.,  96.],
                    [152.,  48.]],

                [[ 80.,  48.],
                    [152.,  48.],
                    [152.,  32.],
                    [ 80.,  32.],
                    [ 80.,  48.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 32.,  32.],
                    [  0.,  32.],
                    [  0., 112.],
                    [ 32., 112.],
                    [ 32.,  32.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]],

                [[ 32.,  96.],
                    [ 48.,  96.],
                    [ 48.,  64.],
                    [ 32.,  64.],
                    [ 32.,  96.],
                    [ -1.,  -1.],
                    [ -1.,  -1.]]
                ]),
            'metrics': np.array([
                2.4320000e+04, 2.1000000e+01, 2.300000e+01, 1.3000000e+01,
                1.0000000e+01, 2.6000000e+01, 6.528000e+03, 5.1200000e+03
            ])
            # [area, num_pins, num_total_nets, num_b2b_nets, 
            # num_p2b_nets, num_hardconstraints, b2b_weighted_wl, p2b_weighted_wl]
        }
        self.pol = [Point(x,y) for x,y in list(self.data['vertex_blocks'][17]) if x != -1 and y != -1]
        self.recs = fun.strop_decomposition(self.pol)
    
    def test_strop_decomposition(self):
        self.assertListEqual(self.recs, [[116.0, 56.0, 72.0, 16.0], [140.0, 80.0, 24.0, 32.0]])

    def test_floorsetinstance_netlist(self, d:float|None=None, t:bool=False):
        fs = FloorSetInstance(self.data, d, t)
        txt_yaml:str = str(fs.write_yaml_FPEF())
        self.assertIsNotNone(txt_yaml)
        Netlist(txt_yaml)

    def test_compute_perimeter(self):
        per = fun.compute_perimeter(self.pol)
        self.assertIsInstance(per, float)
        self.assertAlmostEqual(per, 240., delta=4)

    def test_is_point_inside_polygon(self):
        inside = fun.is_point_inside_polygon(Point(110,80), self.pol)
        self.assertFalse(inside, "Point not inside the polygon")
        inside = fun.is_point_inside_polygon(Point(140,60), self.pol)
        self.assertTrue(inside, "Point is inside the polygon")

    def test_compute_centroid(self):
        c = fun.compute_centroid(self.recs)
        self.assertIsInstance(c,Point)
        self.assertAlmostEqual(c, Point(125.6,65.6), delta=4)


if __name__ == "__main__":
    unittest.main()
