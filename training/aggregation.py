import torch
import logging
from typing import List, Dict, Optional
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

logger = logging.getLogger(__name__)

def validate_models(models: List[Dict[str, torch.Tensor]]) -> bool:
    """Validate model structures match"""
    if not models:
        raise ValueError("Empty model list provided")
    
    base = models[0]
    return all(
        m.keys() == base.keys() and 
        all(m[k].shape == base[k].shape for k in base.keys())
        for m in models
    )

def aggregate_models(
    models: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    clip_value: float = 10.0,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Aggregate multiple models with weighted averaging and gradient clipping
    
    Args:
        models: List of model state dictionaries
        weights: Optional weight for each model
        clip_value: Maximum gradient norm
        device: Target device for computation
    """
    try:
        if not validate_models(models):
            raise ValueError("Inconsistent model structures")

        n_models = len(models)
        weights = weights or [1.0 / n_models] * n_models
        
        if len(weights) != n_models:
            raise ValueError("Weights length must match number of models")
        if not torch.allclose(torch.tensor(weights).sum(), torch.tensor(1.0)):
            raise ValueError("Weights must sum to 1")
            
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aggregated_dict = {}
        
        # Process each parameter with progress bar
        for key in tqdm(models[0].keys(), desc="Aggregating models"):
            try:
                # Move to device and aggregate
                tensors = [m[key].to(device) for m in models]
                weighted_sum = torch.zeros_like(tensors[0])
                
                for w, tensor in zip(weights, tensors):
                    # Clip gradients if needed
                    if tensor.grad is not None:
                        clip_grad_norm_(tensor, clip_value)
                    weighted_sum.add_(tensor * w)
                
                aggregated_dict[key] = weighted_sum
                
            except Exception as e:
                logger.error(f"Error aggregating parameter {key}: {str(e)}")
                raise

        return aggregated_dict
        
    except Exception as e:
        logger.error(f"Model aggregation failed: {str(e)}")
        raise
