"""vAIn training module for production environment."""

__version__ = '0.2.1'

try:
    from .federated_training import FederatedTraining
    from .federated_client import FederatedClient
    # Fix: Import from utils.metrics instead of training.metrics
    from utils.metrics import (
        compute_accuracy as calculate_accuracy,
        compute_loss as calculate_loss
    )
    # Keep importing calculate_data_quality from training.metrics if it exists there
    from .metrics import calculate_data_quality
    
    from .compression import (
        AdaptiveCompression,
        CompressionStats,
        CompressionError
    )
    from .model_optimizer import ModelOptimizer
    from .mobile_optimization import MobileOptimizer
    from .dnc_loss import DNCLoss
except ImportError as e:
    raise ImportError(f"Failed to import required training components: {e}")

__all__ = [
    'FederatedTraining',
    'FederatedClient',
    'calculate_accuracy',
    'calculate_loss',
    'calculate_data_quality',
    'AdaptiveCompression',
    'CompressionStats',
    'CompressionError',
    'ModelOptimizer',
    'MobileOptimizer',
    'DNCLoss'
]
