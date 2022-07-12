import unittest

from tools.hello.hello import hello


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello(), "Hello!")
        self.assertEqual(hello("Marçal"), "Hello Marçal!")


if __name__ == '__main__':
    unittest.main()
