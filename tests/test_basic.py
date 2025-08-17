import unittest

class TestBasic(unittest.TestCase):
    def test_simple(self):
        """A simple test that should always pass"""
        self.assertEqual(1, 1)

def test_imports():
    """Test that basic imports work correctly"""
    import tqdm
    import web3
    assert tqdm.__version__ is not None
    assert web3.__version__ is not None

if __name__ == "__main__":
    unittest.main()