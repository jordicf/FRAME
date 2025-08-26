# (c) Antoni Pech-Alberich 2025
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest
from tools.early_router.build_model import FeedThrough
from tools.early_router.hanan import HananGrid, HananGraph3D, HananCell, manhattan_dist, Layer, HananEdge3D, HananNode3D
from frame.geometry.geometry import Point, Shape
from frame.netlist.netlist import Netlist
from ruamel.yaml import YAML

class TestEarlyRouter(unittest.TestCase):
    def setUp(self):
        data = {
            'Modules':
            {
                'M0':
                {
                    'area': 65.0,
                    'rectangles':
                    [
                        [4.5,6.5,5,13]
                    ]
                },
                'M1':
                {
                    'area': 68.0,
                    'rectangles':
                    [
                        [5,15.5,10,5],
                        [8.5,10,3,6]
                    ]
                },
                'T0':
                {
                    'center': [0.,0.],
                    'io_pin': True
                },
                'T1':
                {
                    'center': [3.,18.],
                    'io_pin': True
                }
            },
            'Nets':
            [
                ['M0', 'M1', 1024.],
                ['T0', 'M0', 'M1', 256.],
                ['T1', 'M0', 256.]
            ]
        }
        yaml_txt = YAML.dump(data, sort_keys=False)
        self.fp_with_blanks = Netlist(yaml_txt)
        #from tools.draw.draw import get_floorplan_plot
        #get_floorplan_plot(self.fp_with_blanks, Shape(10,18)).save("./tests/early_router/test.png")

        self.hn_ref = HananGrid(
            [HananCell(_id=(1, 0), center=Point(x=4.5, y=3.5), width_capacity=5.0, height_capacity=7.0, modulename='M0'),
                HananCell(_id=(1, 1), center=Point(x=4.5, y=10.0), width_capacity=5.0, height_capacity=6.0, modulename='M0'),
                HananCell(_id=(0, 2), center=Point(x=1.0, y=15.5), width_capacity=2.0, height_capacity=5.0, modulename='M1'),
                HananCell(_id=(1, 2), center=Point(x=4.5, y=15.5), width_capacity=5.0, height_capacity=5.0, modulename='M1'),
                HananCell(_id=(2, 2), center=Point(x=8.5, y=15.5), width_capacity=3.0, height_capacity=5.0, modulename='M1'),
                HananCell(_id=(2, 1), center=Point(x=8.5, y=10.0), width_capacity=3.0, height_capacity=6.0, modulename='M1')]
            )
    
    def test_hanangrid(self):
        hn = HananGrid(self.fp_with_blanks)
        self.assertEqual(hn, self.hn_ref)
        
    def test_get_closest_cell_to_point(self):
        cell = self.hn_ref.get_closest_cell_to_point(Point(0.,0.))
        self.assertEqual(cell, HananCell(_id=(1, 0), center=Point(x=4.5, y=3.5), width_capacity=5.0, height_capacity=7.0, modulename='M0'))
        self.hn_ref.add_blank_cells()
        cell = self.hn_ref.get_closest_cell_to_point(Point(0.,0.))
        self.assertEqual(cell, HananCell(_id=(0, 0), center=Point(x=1., y=3.5), width_capacity=2.0, height_capacity=7.0, modulename=''))

    def test_hanangraph3D(self):
        layers = [Layer('H', pitch=76), Layer('V', pitch=76)]
        options = {'asap7': True}
        hg = HananGraph3D(self.hn_ref, layers, netlist=self.fp_with_blanks, **options)
        e = HananEdge3D(
            source=HananNode3D(
                _id=(1, 0, 1), 
                center=Point(x=4.5, y=3.5), 
                modulename='M0'
                ), 
            target=HananNode3D(
                _id=(1, 1, 1), 
                center=Point(x=4.5, y=10.0), 
                modulename='M0'
                ),
            length=6.5,
            capacity=65, 
            crossing=False, 
            via=False
            )
        self.assertEqual(hg.adjacent_list[(1,0,1)][(1,1,1)], e)

    def test_manhattan_dist(self):
        self.assertAlmostEqual(manhattan_dist(Point(3, 4),Point(8, 5)), 6., delta=6)
        self.assertAlmostEqual(manhattan_dist(Point(6.3, 1.4),Point(1.3, 2.6)), 6.2, delta=6)

    def test_solve(self):
        options = {'asap7': True, 'add_blank':True, 'pitch_layers': (6,6)}
        ft = FeedThrough(self.fp_with_blanks, **options)
        ft.add_nets(self.fp_with_blanks.edges)
        ft.build(verbose=False)
        sucess, _ = ft.solve(f_wl=0.1, f_mc=0.2, f_vu=0.7, verbose=False)
        self.assertTrue(sucess)

if __name__ == "__main__":
    unittest.main()
