import torch
from .federated_client import FederatedClient
from .aggregation import aggregate_models

class FederatedTraining:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.clients = []
        
    def train(self):
        for round in range(self.config.num_rounds):
            # Select clients
            active_clients = self._select_clients()
            
            # Train on each client
            client_models = []
            for client in active_clients:
                client_models.append(client.train())
            
            # Aggregate models
            self.model = aggregate_models(client_models)
