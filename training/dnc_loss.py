import torch
import torch.nn as nn

class DNCLoss(nn.Module):
    def __init__(self, memory_reg_factor: float = 0.1):
        super().__init__()
        self.memory_reg_factor = memory_reg_factor
        self.task_loss = nn.CrossEntropyLoss()
        
    def forward(self, transformer_out, read_vector, target, memory):
        # Task-specific loss
        task_loss = self.task_loss(transformer_out[-1], target)
        
        # Memory usage regularization
        memory_sparsity = torch.norm(memory, p=1)
        memory_coherence = torch.norm(torch.matmul(memory, memory.t()) - torch.eye(memory.size(0)))
        
        # Combined loss
        total_loss = task_loss + self.memory_reg_factor * (memory_sparsity + memory_coherence)
        
        return total_loss
