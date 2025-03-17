import torch
from torch import nn
from typing import Dict
import numpy as np

def calculate_data_quality(model: nn.Module) -> float:
    """Calculate data quality score based on model metrics"""
    quality_metrics = {
        'gradient_noise': _calculate_gradient_noise(model),
        'param_variance': _calculate_param_variance(model),
        'update_magnitude': _calculate_update_magnitude(model)
    }
    
    # Weighted combination of metrics
    weights = {
        'gradient_noise': 0.4,
        'param_variance': 0.3,
        'update_magnitude': 0.3
    }
    
    score = sum(metric * weights[name] 
                for name, metric in quality_metrics.items())
    return float(np.clip(score, 0, 1))

def _calculate_gradient_noise(model: nn.Module) -> float:
    """Estimate gradient noise using param statistics"""
    total_variance = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            total_variance += torch.var(param.grad.data).item()
            param_count += 1
            
    return 1.0 / (1.0 + total_variance / max(1, param_count))

# ...Add other metric calculation functions...
