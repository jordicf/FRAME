import unittest

from frame.allocation.allocation import Allocation

class MyTestCase(unittest.TestCase):

    def test_read_allocation(self):
        a = Allocation(alloc1)
        a.write_yaml("alloc1_w.yml")


if __name__ == '__main__':
    unittest.main()

alloc1 = """
[ 
[[5,3,10,20], {M1: 0.5, M2: 0.3}],
[[6,5,1,8.5], {M2: 0.2, M3: 0.4, M5: 0.2}]
]
"""
