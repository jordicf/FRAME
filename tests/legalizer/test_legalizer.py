import unittest
import sys
from pathlib import Path
from tools.legalizer import legalizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]  
BENCH_EXAM_DIR = PROJECT_ROOT / 'tools' / 'legalizer' / 'bench-exam'

DIE_PATH = BENCH_EXAM_DIR / 'test_die.yaml'
NETLIST_PATH = BENCH_EXAM_DIR / 'test_netlist.yaml'

class TestLegalizerIntegration(unittest.TestCase):
    def test_legalizer_main_runs(self):
        # Patch sys.argv to simulate command-line usage
        argv_backup = sys.argv.copy()
        sys.argv = [
            'legalizer.py',
            str(NETLIST_PATH),
            str(DIE_PATH),
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