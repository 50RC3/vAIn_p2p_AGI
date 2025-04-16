import unittest

class TestBasic(unittest.TestCase):
    def test_simple(self):
        """A simple test that should always pass"""
        self.assertEqual(1, 1)

if __name__ == "__main__":
    unittest.main()