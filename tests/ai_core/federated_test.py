import unittest
import torch
from training.federated import FederatedLearning
from models.simple_nn import SimpleNN
from config import Config

class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.model = SimpleNN(self.config)
        self.federated = FederatedLearning(self.config)
        
    def test_client_selection(self):
        num_clients = 10
        for i in range(num_clients):
            self.federated.register_client(None)  # Mock client
        selected = self.federated.select_clients()
        self.assertEqual(len(selected), self.config.clients_per_round)

if __name__ == '__main__':
    unittest.main()
