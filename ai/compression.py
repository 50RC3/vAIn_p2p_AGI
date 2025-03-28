# ai/compression.py
"""Module for gradient compression techniques in federated learning."""
import torch
from typing import Dict, Any, Optional

def compress_gradients(gradients: Dict[str, torch.Tensor],
                      compression_rate: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Compress gradients using top-k sparsification.
    
    Args:
        gradients: Dictionary of parameter name to gradient tensor
        compression_rate: Fraction of gradients to keep (0-1)
        
    Returns:
        Dictionary of compressed gradients
    """
    compressed = {}
    for name, grad in gradients.items():
        if grad is None:
            compressed[name] = None
            continue
            
        # Flatten the gradient
        flat_grad = grad.view(-1)
        
        # Determine k based on compression rate
        k = max(1, int(flat_grad.numel() * compression_rate))
        
        # Get the indices of the top k elements by magnitude
        _, indices = torch.topk(torch.abs(flat_grad), k)
        
        # Create a sparse tensor with the same shape as the original
        sparse_grad = torch.zeros_like(flat_grad)
        sparse_grad[indices] = flat_grad[indices]
        
        # Reshape back to original shape
        compressed[name] = sparse_grad.view_as(grad)
    
    return compressed

def decompress_gradients(compressed_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Decompress gradients - identity function as our compression is lossy.
    
    Args:
        compressed_gradients: Dictionary of compressed gradients
        
    Returns:
        Dictionary of decompressed gradients
    """
    return compressed_gradients