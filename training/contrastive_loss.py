import torch
import torch.nn.functional as F
from typing import Union, Optional
import warnings

class ContrastiveLoss(torch.nn.Module):
    """Production-ready Contrastive Loss implementation with validation and stability"""
    
    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self._validate_temperature(temperature)
        self.temperature = temperature
        self.eps = eps
        
    def _validate_temperature(self, temp: float) -> None:
        """Validate temperature parameter"""
        if not isinstance(temp, (int, float)):
            raise TypeError(f"Temperature must be a number, got {type(temp)}")
        if temp <= 0:
            raise ValueError(f"Temperature must be positive, got {temp}")
            
    def _validate_inputs(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate input tensors"""
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Features must be a torch.Tensor, got {type(features)}")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Labels must be a torch.Tensor, got {type(labels)}")
            
        if features.dim() != 2:
            raise ValueError(f"Features must be 2D (batch_size, features), got shape {features.shape}")
        if labels.dim() != 1:
            raise ValueError(f"Labels must be 1D (batch_size), got shape {labels.shape}")
            
        if len(features) != len(labels):
            raise ValueError(f"Batch size mismatch: features {len(features)} vs labels {len(labels)}")
            
        if len(features) < 2:
            raise ValueError("Batch size must be at least 2 for contrastive loss")
            
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with full validation and numerical stability"""
        self._validate_inputs(features, labels)
        
        device = features.device
        if labels.device != device:
            labels = labels.to(device)
            
        batch_size = features.shape[0]
        
        # Normalize features with stability
        features = F.normalize(features, dim=1, eps=self.eps)
        
        # Compute similarity matrix with stability
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0/self.temperature, max=1.0/self.temperature)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create masks for positive and negative pairs
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = labels_matrix.float()
        negative_mask = (~labels_matrix).float()
        
        # Check if we have any positive pairs
        if not torch.any(positive_mask):
            warnings.warn("No positive pairs found in batch", RuntimeWarning)
            return torch.tensor(0.0, device=device)
            
        # Compute loss with numerical stability
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)
        
        loss = -(positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + self.eps)
        
        # Average only over samples that have positives
        num_valid = torch.sum(positive_mask.sum(1) > 0)
        loss = loss.sum() / (num_valid + self.eps)
        
        return loss
