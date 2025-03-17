import torch
import torch.nn as nn
from typing import Dict, List

class ReptileModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner_lr = config.inner_learning_rate
        self.feature_extractor = nn.Sequential(
            nn.Linear(784, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def adapt(self, support_set: List[torch.Tensor], num_steps: int) -> Dict[str, torch.Tensor]:
        adapted_state = {k: v.clone() for k, v in self.state_dict().items()}
        optimizer = torch.optim.SGD(self.parameters(), lr=self.inner_lr)
        
        for _ in range(num_steps):
            for x, y in support_set:
                optimizer.zero_grad()
                loss = self.compute_loss(x, y)
                loss.backward()
                optimizer.step()
                
        return adapted_state
