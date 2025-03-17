import unittest
import torch
from models.simple_nn import SimpleNN
from config import Config

class TestSimpleNN(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.model = SimpleNN(self.config)
        
    def test_forward_pass(self):
        batch_size = 32
        input_dim = 784
        x = torch.randn(batch_size, input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size, 10))

if __name__ == '__main__':
    unittest.main()
