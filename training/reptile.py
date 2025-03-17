import torch
from typing import List, Dict
from models.reptile_meta.reptile_model import ReptileModel

class ReptileTrainer:
    def __init__(self, model: ReptileModel, 
                 inner_lr: float = 0.01, 
                 outer_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        
    def adapt_to_task(self, support_set: List[torch.Tensor], 
                      query_set: List[torch.Tensor], 
                      num_inner_steps: int = 5):
        # Store original parameters
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Adapt to task
        adapted_state = self.model.adapt(support_set, num_inner_steps)
        
        # Compute meta gradient
        meta_grads = {k: orig_state[k] - adapted_state[k] 
                     for k in orig_state.keys()}
        
        # Update model parameters
        for k, v in self.model.state_dict().items():
            v.data.add_(meta_grads[k], alpha=-self.outer_lr)
