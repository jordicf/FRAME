import unittest
from pathlib import Path
from tools.legalizer.legalizer import compute_options, Model

PROJECT_ROOT = Path(__file__).resolve().parents[2]  #back to the FRAME
BENCH_EXAM_DIR = PROJECT_ROOT / 'tools' / 'legalizer' / 'bench-exam'  #locate the bench-exam directory

DIE_PATH = BENCH_EXAM_DIR / 'test_die.yaml'
NETLIST_PATH = BENCH_EXAM_DIR / 'test_netlist.yaml'
class TestLegalizerUnit(unittest.TestCase):
    def setUp(self):
        self.options = {
            'netlist': str(NETLIST_PATH),
            'die': str(DIE_PATH),
            'max_ratio': 3.0,
            'num_iter': 2,
            'radius': 1,
            'wl_mult': 1,
            'otol_initial': 1e-2,
            'otol_final': 1e-4,
            'rtol_initial': 1e-2,
            'rtol_final': 1e-4,
            'tol_decay': 0.5,
            'palette_seed': None,
            'tau_initial': None,
            'tau_decay': 0.3,
            'file': None,
            'plot': False,
            'small_steps': False,
            'verbose': False,
        }

    def test_compute_options(self):
        ml, al, xl, yl, wl, hl, die_width, die_height, hyper, max_ratio, og_names, terminal_map = compute_options(self.options)
        self.assertGreater(len(ml), 0)
        self.assertGreater(die_width, 0)
        self.assertGreater(len(og_names), 0)
        self.assertIsInstance(hyper, list)

    def test_model_objective_and_solve(self):
        ml, al, xl, yl, wl, hl, die_width, die_height, hyper, max_ratio, og_names, terminal_map = compute_options(self.options)
        model = Model(
            ml, al, xl, yl, wl, hl, die_width, die_height, hyper, max_ratio, og_names,
            self.options['otol_initial'], self.options['tol_decay'], self.options['wl_mult'],
            tau_initial=self.options['tau_initial'],
            tau_decay=self.options['tau_decay'],
            otol_initial=self.options['otol_initial'],
            otol_final=self.options['otol_final'],
            rtol_initial=self.options['rtol_initial'],
            rtol_final=self.options['rtol_final'],
            tol_decay=self.options['tol_decay'],
            terminal_map=terminal_map,
            palette_seed=self.options['palette_seed'],
        )
        obj = model.objective()
        self.assertIsNotNone(obj)
        self.assertTrue(hasattr(obj, 'evaluate'))
        model.solve(verbose=False, small_steps=False, radius=1)
        netlist = model.get_netlist()
        self.assertTrue(hasattr(netlist, 'modules'))
        self.assertTrue(hasattr(netlist, 'edges'))
        self.assertGreater(len(netlist.modules), 0)

if __name__ == '__main__':
    unittest.main() 