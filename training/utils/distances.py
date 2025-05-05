"""Model distance utilities for federated learning"""

import torch
import torch.nn as nn
from typing import List

def compute_pairwise_distances(models: List[nn.Module]) -> torch.Tensor:
    """Compute pairwise distances between model parameters
    
    Args:
        models: List of PyTorch models
        
    Returns:
        torch.Tensor: Pairwise distance matrix
    """
    if not models:
        return torch.tensor([])
        
    n_models = len(models)
    distances = torch.zeros((n_models, n_models))
    
    # Extract parameters for each model
    param_vectors = []
    for model in models:
        params = []
        for param in model.parameters():
            params.append(param.detach().view(-1))
        param_vectors.append(torch.cat(params))
    
    # Compute pairwise distances
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Euclidean distance between parameter vectors
            dist = torch.norm(param_vectors[i] - param_vectors[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
    return distances