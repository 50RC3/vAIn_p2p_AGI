import torch
from torch import nn
from typing import Dict, Optional
import numpy as np
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def safe_calculation():
    """Context manager for safe metric calculation with error handling"""
    try:
        yield
    except RuntimeError as e:
        logger.error(f"Torch runtime error in metric calculation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in metric calculation: {e}")
        raise

def calculate_data_quality(model: nn.Module) -> float:
    """
    Calculate data quality score based on model metrics
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        float: Quality score between 0 and 1
        
    Raises:
        ValueError: If model is None or has no parameters
        RuntimeError: If calculation fails due to invalid model state
    """
    if model is None:
        raise ValueError("Model cannot be None")
        
    if not any(True for _ in model.parameters()):
        raise ValueError("Model has no parameters")

    with safe_calculation():
        quality_metrics = {
            'gradient_noise': _calculate_gradient_noise(model),
            'param_variance': _calculate_param_variance(model), 
            'update_magnitude': _calculate_update_magnitude(model)
        }
        
        weights = {
            'gradient_noise': 0.4,
            'param_variance': 0.3,
            'update_magnitude': 0.3
        }
        
        score = sum(metric * weights[name] 
                    for name, metric in quality_metrics.items())
                    
        return float(np.clip(score, 0, 1))

def _calculate_gradient_noise(model: nn.Module) -> float:
    """
    Estimate gradient noise using parameter statistics
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        float: Normalized noise score between 0 and 1
    """
    with safe_calculation():
        total_variance = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_var = torch.var(param.grad.data).item()
                # Handle numerical stability
                if not np.isnan(grad_var):
                    total_variance += grad_var
                    param_count += 1
                    
        return 1.0 / (1.0 + total_variance / max(1, param_count))

def _calculate_param_variance(model: nn.Module) -> float:
    """
    Calculate variance of model parameters
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        float: Normalized variance score between 0 and 1
    """
    with safe_calculation():
        total_var = 0.0
        count = 0
        
        for param in model.parameters():
            param_var = torch.var(param.data).item()
            if not np.isnan(param_var):
                total_var += param_var
                count += 1
                
        # Normalize using softmax-style function
        return 1.0 / (1.0 + np.exp(total_var / max(1, count) - 1.0))

def _calculate_update_magnitude(model: nn.Module) -> float:
    """
    Calculate magnitude of parameter updates
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        float: Normalized magnitude score between 0 and 1
    """
    with safe_calculation():
        total_magnitude = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                update_magnitude = torch.norm(param.grad.data).item()
                if not np.isnan(update_magnitude):
                    total_magnitude += update_magnitude
                    param_count += 1
                    
        # Normalize using exponential decay
        return np.exp(-total_magnitude / max(1, param_count))
