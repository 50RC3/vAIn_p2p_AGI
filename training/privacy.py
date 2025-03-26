import torch
from typing import Optional

def add_differential_privacy_noise(
    tensor: torch.Tensor,
    epsilon: float,
    delta: float,
    sensitivity: Optional[float] = None
) -> torch.Tensor:
    """Add Gaussian noise for differential privacy."""
    if sensitivity is None:
        sensitivity = torch.norm(tensor).item()
    
    # Calculate noise scale using Gaussian mechanism
    noise_scale = (sensitivity / epsilon) * (2 * torch.log(1.25 / delta)).sqrt()
    
    # Generate and add noise
    noise = torch.randn_like(tensor) * noise_scale
    return tensor + noise
