"""
Module for compression techniques used in federated learning.
"""
import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Iterator, Union

def compress_gradients(gradients: Union[Dict[str, torch.Tensor], List[torch.Tensor]], 
                       compression_ratio: float = 0.1, 
                       quantize: bool = True) -> Dict[str, Any]:
    """
    Compress gradients for efficient transmission in federated learning.
    
    Args:
        gradients: Dictionary of parameter name to gradient tensor or list of gradient tensors
        compression_ratio: Fraction of gradients to keep (0.1 = top 10%)
        quantize: Whether to quantize values to reduce precision
        
    Returns:
        Compressed representation of gradients
    """
    compressed = {}
    
    if isinstance(gradients, dict):
        for name, grad in gradients.items():
            if grad is None or not torch.is_tensor(grad):
                compressed[name] = grad
                continue
                
            # Convert to numpy for easier handling
            grad_np = grad.detach().cpu().numpy()
            
            # Sparsification - keep only top k% of values by magnitude
            if compression_ratio < 1.0:
                size = grad_np.size
                k = max(1, int(size * compression_ratio))
                
                # Flatten the tensor
                flattened = grad_np.flatten()
                
                # Find indices of top k values by magnitude
                indices = np.argsort(np.abs(flattened))[-k:]
                values = flattened[indices]
                
                # Quantize values if requested
                if quantize:
                    # Simple 16-bit quantization
                    values = values.astype(np.float16).astype(np.float32)
                
                # Store as sparse representation
                compressed[name] = {
                    'shape': grad_np.shape,
                    'indices': indices.tolist(),
                    'values': values.tolist(),
                    'quantized': quantize
                }
            else:
                # Full gradient with optional quantization
                if quantize:
                    grad_np = grad_np.astype(np.float16).astype(np.float32)
                
                compressed[name] = {
                    'shape': grad_np.shape,
                    'data': grad_np.tolist(),
                    'quantized': quantize
                }
    elif isinstance(gradients, list):
        compressed = {'list_data': []}
        for i, grad in enumerate(gradients):
            if grad is None or not torch.is_tensor(grad):
                compressed['list_data'].append(grad)
                continue
                
            # Apply the same logic as above
            grad_np = grad.detach().cpu().numpy()
            
            if compression_ratio < 1.0:
                size = grad_np.size
                k = max(1, int(size * compression_ratio))
                flattened = grad_np.flatten()
                indices = np.argsort(np.abs(flattened))[-k:]
                values = flattened[indices]
                
                if quantize:
                    values = values.astype(np.float16).astype(np.float32)
                
                compressed['list_data'].append({
                    'shape': grad_np.shape,
                    'indices': indices.tolist(),
                    'values': values.tolist(),
                    'quantized': quantize
                })
            else:
                if quantize:
                    grad_np = grad_np.astype(np.float16).astype(np.float32)
                
                compressed['list_data'].append({
                    'shape': grad_np.shape,
                    'data': grad_np.tolist(),
                    'quantized': quantize
                })
    
    return compressed

def decompress_gradients(compressed_grads: Dict[str, Any]) -> Union[Dict[str, torch.Tensor], List[torch.Tensor]]:
    """
    Decompress gradients received during federated learning.
    
    Args:
        compressed_grads: Compressed representation from compress_gradients
        
    Returns:
        Dictionary mapping parameter names to gradient tensors or list of gradient tensors
    """
    decompressed = {}
    
    # Check if it's a list-based compression
    if 'list_data' in compressed_grads:
        result = []
        for item in compressed_grads['list_data']:
            if isinstance(item, dict) and 'shape' in item:
                shape = item['shape']
                
                if 'indices' in item:  # Sparse representation
                    # Initialize with zeros
                    grad_np = np.zeros(np.prod(shape))
                    
                    # Put values back in their original positions
                    indices = item['indices']
                    values = item['values']
                    
                    for idx, val in zip(indices, values):
                        grad_np[idx] = val
                    
                    # Reshape back to original dimensions
                    grad_np = grad_np.reshape(shape)
                else:  # Full representation
                    grad_np = np.array(item['data']).reshape(shape)
                
                # Convert back to PyTorch tensor
                result.append(torch.tensor(grad_np))
            else:
                result.append(item)  # Keep as is (None or non-tensor)
        
        return result
    else:
        # Dictionary-based compression
        for name, item in compressed_grads.items():
            if isinstance(item, dict) and 'shape' in item:
                shape = item['shape']
                
                if 'indices' in item:  # Sparse representation
                    # Initialize with zeros
                    grad_np = np.zeros(np.prod(shape))
                    
                    # Put values back in their original positions
                    indices = item['indices']
                    values = item['values']
                    
                    for idx, val in zip(indices, values):
                        grad_np[idx] = val
                    
                    # Reshape back to original dimensions
                    grad_np = grad_np.reshape(shape)
                else:  # Full representation
                    grad_np = np.array(item['data']).reshape(shape)
                
                # Convert back to PyTorch tensor
                decompressed[name] = torch.tensor(grad_np)
            else:
                decompressed[name] = item  # Keep as is (None or non-tensor)
    
    return decompressed
