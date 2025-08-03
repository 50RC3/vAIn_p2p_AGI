"""
Utility functions for the vAIn P2P AGI system.
"""

try:
    from .metrics import compute_accuracy, compute_loss
except ImportError:
    compute_accuracy = None
    compute_loss = None

try:
    from .helpers import save_checkpoint, load_checkpoint
except ImportError:
    save_checkpoint = None
    load_checkpoint = None

__all__ = [
    'compute_accuracy',
    'compute_loss',
    'save_checkpoint',
    'load_checkpoint'
]
