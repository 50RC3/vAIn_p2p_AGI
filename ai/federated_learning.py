import torch
from typing import Dict, List, Tuple
import numpy as np
from .compression import compress_gradients, decompress_gradients

class FederatedLearner:
    def __init__(self, model: torch.nn.Module, config: Dict):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.min_clients = config.get('min_clients', 2)
        self.compression_rate = config.get('compression_rate', 0.01)
        self.error_feedback = {}
        
    async def aggregate_models(self, model_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Hierarchical FedAvg aggregation with compression"""
        if len(model_updates) < self.min_clients:
            raise ValueError("Not enough clients for aggregation")
            
        # Decompress updates and apply error feedback
        decompressed_updates = []
        for update in model_updates:
            update_with_feedback = self._apply_error_feedback(update)
            decompressed = decompress_gradients(update_with_feedback)
            decompressed_updates.append(decompressed)

        # Aggregate hierarchically by layer
        aggregated_model = {}
        for key in decompressed_updates[0].keys():
            aggregated_model[key] = self._hierarchical_aggregate(
                [update[key] for update in decompressed_updates]
            )
        
        return aggregated_model

    def _hierarchical_aggregate(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate tensors hierarchically in groups"""
        if len(tensors) <= 10:  # Base case
            return torch.mean(torch.stack(tensors), dim=0)
            
        # Split into groups and aggregate recursively
        groups = np.array_split(tensors, max(2, len(tensors)//10))
        group_results = [self._hierarchical_aggregate(g.tolist()) for g in groups]
        return torch.mean(torch.stack(group_results), dim=0)
        
    async def train_round(self, local_data, epochs=1):
        """Local model training"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for _ in range(epochs):
            for batch in local_data:
                optimizer.zero_grad()
                loss = self.model.training_step(batch)
                loss.backward()
                optimizer.step()
                
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
