import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Tuple, Optional

class MetaReptile:
    """Meta-Reptile implementation for DNC-based models."""
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float = 0.01,
                 meta_lr: float = 0.001,
                 num_inner_steps: int = 5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.task_loss = nn.CrossEntropyLoss()

    def inner_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[nn.Module, float]:
        """Perform inner loop updates on a task."""
        temp_model = copy.deepcopy(self.model)
        inner_optimizer = optim.SGD(temp_model.parameters(), lr=self.inner_lr)
        
        final_loss = 0.0
        for _ in range(self.num_inner_steps):
            inner_optimizer.zero_grad()
            transformer_out, read_vector = temp_model(x)
            loss = self.task_loss(transformer_out[-1], y)
            loss.backward()
            inner_optimizer.step()
            final_loss = loss.item()

        return temp_model, final_loss

    def meta_update(self, temp_model: nn.Module) -> None:
        """Update meta-parameters using Reptile update rule."""
        for meta_param, temp_param in zip(self.model.parameters(), temp_model.parameters()):
            if meta_param.grad is None:
                meta_param.grad = torch.zeros_like(meta_param.data)
            meta_param.grad.data.add_((temp_param.data - meta_param.data) / self.meta_lr)

    def adapt_to_task(self, support_x: torch.Tensor, support_y: torch.Tensor) -> float:
        """Adapt model to new task using support set."""
        self.meta_optimizer.zero_grad()
        temp_model, loss = self.inner_update(support_x, support_y)
        self.meta_update(temp_model)
        self.meta_optimizer.step()
        return loss

    def evaluate_task(self, query_x: torch.Tensor, query_y: torch.Tensor) -> Tuple[float, float]:
        """Evaluate model on query set after adaptation."""
        with torch.no_grad():
            transformer_out, _ = self.model(query_x)
            loss = self.task_loss(transformer_out[-1], query_y)
            pred = transformer_out[-1].argmax(dim=1)
            accuracy = (pred == query_y).float().mean()
        return loss.item(), accuracy.item()
