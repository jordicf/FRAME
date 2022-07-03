import unittest

from utils import valid_identifier, string_is_number, is_number


class TestUtils(unittest.TestCase):
    def test_valid_identifiers(self):
        valid = ['_this_is_valid_9999', 'This_Is_Valid', '_']
        not_valid = ['0_this_is_not_valid_9999', 'strange.not.valid', '']
        for s in valid:
            self.assertTrue(valid_identifier(s))
        for s in not_valid:
            self.assertFalse(valid_identifier(s))

    def test_numbers(self):
        numbers = ['0000', '3', '5.6', '4e-5', '8E6', '-45.47']
        not_numbers = ['hello', '', '123a']
        for s in numbers:
            self.assertTrue(string_is_number(s))
        for s in not_numbers:
            self.assertFalse(string_is_number(s))

    def test_type_numbers(self):
        numbers = [0, 3, -5, 38.6, -4e+7, 1e-4]
        not_numbers = ['hello', '38', [1, 2, 3], (2, 5)]
        for x in numbers:
            self.assertTrue(is_number(x))
        for x in not_numbers:
            self.assertFalse(is_number(x))


if __name__ == '__main__':
    unittest.main()
