import torch
import torch.optim as optim
from models.hybrid_memory_system import HybridMemorySystem
from .meta_reptile import MetaReptile

class Agent:
    def __init__(self, config):
        try:
            self.model = HybridMemorySystem(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                memory_vector_dim=config.memory_vector_dim,
                nhead=config.num_heads,
                num_layers=config.num_layers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HybridMemorySystem: {e}")
            
        self.meta_reptile = MetaReptile(
            self.model,
            inner_lr=config.inner_learning_rate,
            meta_lr=config.meta_learning_rate
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)

    def local_update(self, x: torch.Tensor, y: torch.Tensor, meta_steps: int):
        if meta_steps <= 0:
            raise ValueError("meta_steps must be positive")
            
        try:
            loss = 0
            for _ in range(meta_steps):
                loss = self.meta_reptile.adapt_to_task(x, y)
            return loss
        except Exception as e:
            raise RuntimeError(f"Error during local update: {e}")
