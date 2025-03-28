import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LossComponents:
    task_loss: float
    memory_sparsity: float 
    memory_coherence: float
    total_loss: float

class DNCLossError(Exception):
    pass

class DNCLoss(nn.Module):
    """Differentiable Neural Computer Loss with memory regularization.
    
    Args:
        memory_reg_factor: Factor for memory regularization (default: 0.1)
        eps: Small value for numerical stability (default: 1e-8)
        max_memory_size: Maximum allowed memory size (default: 1e6)
    """
    def __init__(self, memory_reg_factor: float = 0.1, eps: float = 1e-8, 
                 max_memory_size: int = int(1e6)):
        super().__init__()
        self._validate_params(memory_reg_factor, eps, max_memory_size)
        self.memory_reg_factor = memory_reg_factor
        self.eps = eps
        self.max_memory_size = max_memory_size
        self.task_loss = nn.CrossEntropyLoss(reduction='mean')
        self.components = LossComponents(0.0, 0.0, 0.0, 0.0)
        logger.info(f"Initialized DNCLoss with reg_factor={memory_reg_factor}")

    def _validate_params(self, reg_factor: float, eps: float, max_size: int) -> None:
        """Validate initialization parameters"""
        if not 0 < reg_factor <= 1.0:
            raise ValueError("memory_reg_factor must be between 0 and 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if max_size <= 0:
            raise ValueError("max_memory_size must be positive")

    def _validate_inputs(self, transformer_out: torch.Tensor, read_vector: torch.Tensor,
                        target: torch.Tensor, memory: torch.Tensor) -> None:
        """Validate input tensors"""
        if not all(torch.is_tensor(x) for x in [transformer_out, read_vector, target, memory]):
            raise DNCLossError("All inputs must be torch.Tensor")
        if memory.numel() > self.max_memory_size:
            raise DNCLossError(f"Memory size {memory.numel()} exceeds maximum {self.max_memory_size}")
        if len(transformer_out) < 1:
            raise DNCLossError("Empty transformer output")
        if target.numel() == 0:
            raise DNCLossError("Empty target tensor")

    def _compute_memory_regularization(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute memory regularization terms with gradient scaling"""
        # Use memory efficient computation for large matrices
        if memory.numel() > 1e5:
            return self._compute_large_memory_reg(memory)
        
        # Regular computation for smaller matrices
        memory_sparsity = torch.norm(memory, p=1)
        identity = torch.eye(memory.size(0), device=memory.device)
        memory_coherence = torch.norm(torch.matmul(memory, memory.t()) - identity)
        
        return memory_sparsity, memory_coherence

    def _compute_large_memory_reg(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficient computation for large memory matrices"""
        memory_sparsity = torch.norm(memory, p=1)
        # Compute coherence in chunks to save memory
        chunk_size = 1000
        memory_coherence = torch.tensor(0., device=memory.device)
        for i in range(0, memory.size(0), chunk_size):
            end = min(i + chunk_size, memory.size(0))
            chunk = memory[i:end]
            memory_coherence += torch.norm(torch.matmul(chunk, memory.t()) - 
                                        torch.eye(end-i, device=memory.device))
        return memory_sparsity, memory_coherence

    def forward(self, transformer_out: torch.Tensor, read_vector: torch.Tensor,
                target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Compute combined DNC loss with memory regularization."""
        try:
            self._validate_inputs(transformer_out, read_vector, target, memory)

            # Task-specific loss with error checking
            task_loss = self.task_loss(transformer_out[-1], target)
            if torch.isnan(task_loss):
                raise DNCLossError("Task loss computation resulted in NaN")

            # Memory regularization
            memory_sparsity, memory_coherence = self._compute_memory_regularization(memory)
            
            # Combine losses with numerical stability
            total_loss = task_loss + self.memory_reg_factor * (
                memory_sparsity + memory_coherence + self.eps)

            # Store components for monitoring
            self.components = LossComponents(
                task_loss.item(),
                memory_sparsity.item(),
                memory_coherence.item(),
                total_loss.item()
            )

            return total_loss

        except Exception as e:
            logger.error(f"Loss computation failed: {str(e)}")
            raise DNCLossError(f"Loss computation failed: {str(e)}")

    async def compute_loss_interactive(self, transformer_out: torch.Tensor, 
                                     read_vector: torch.Tensor,
                                     target: torch.Tensor, 
                                     memory: torch.Tensor) -> torch.Tensor:
        """Interactive loss computation with progress tracking"""
        try:
            self._validate_inputs(transformer_out, read_vector, target, memory)

            # Track initial memory state
            initial_state = {
                'transformer_out': transformer_out.detach().clone(),
                'memory': memory.detach().clone()
            }

            # Compute core loss components
            task_loss = self.task_loss(transformer_out[-1], target)
            if torch.isnan(task_loss):
                raise DNCLossError("Task loss computation resulted in NaN")

            memory_sparsity, memory_coherence = self._compute_memory_regularization(memory)
            
            # Store loss components for monitoring
            self.components = LossComponents(
                task_loss=task_loss.item(),
                memory_sparsity=memory_sparsity.item(),
                memory_coherence=memory_coherence.item(),
                total_loss=(task_loss + self.memory_reg_factor * 
                          (memory_sparsity + memory_coherence + self.eps)).item()
            )

            return self.components.total_loss

        except Exception as e:
            logger.error(f"Interactive loss computation failed: {str(e)}")
            # Try to recover initial state
            if 'initial_state' in locals():
                transformer_out.data.copy_(initial_state['transformer_out'])
                memory.data.copy_(initial_state['memory'])
            raise DNCLossError(f"Loss computation failed: {str(e)}")
