import unittest
import os
import sys
from tools.legalizer import legalizer

# Get the absolute path to the FRAME root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
EXAMPLE_DIR = os.path.join(ROOT_DIR, 'tools/legalizer/bench-exam')
NETLIST_PATH = os.path.join(EXAMPLE_DIR, 'test_netlist.yaml')
DIE_PATH = os.path.join(EXAMPLE_DIR, 'test_die.yaml')

class TestLegalizerIntegration(unittest.TestCase):
    def test_legalizer_main_runs(self):
        # Patch sys.argv to simulate command-line usage
        argv_backup = sys.argv.copy()
        sys.argv = [
            'legalizer.py',
            NETLIST_PATH,
            DIE_PATH,
            '--num_iter', '1',  # keep it fast
            '--max_ratio', '3.0',
            '--radius', '1',
            '--wl_mult', '1',
            '--otol_initial', '1e-2',
            '--otol_final', '1e-4',
            '--rtol_initial', '1e-2',
            '--rtol_final', '1e-4',
            '--tol_decay', '0.5',
        ]
        try:
            # Should not raise
            result = legalizer.main()
            self.assertEqual(result, 0)
        finally:
            sys.argv = argv_backup

if __name__ == '__main__':
    unittest.main() 