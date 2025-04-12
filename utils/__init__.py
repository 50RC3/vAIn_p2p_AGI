from .metrics import compute_accuracy, compute_loss
from .helpers import save_checkpoint, load_checkpoint

# Import resource monitoring utilities
try:
    import psutil
    import torch
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    import logging
    logging.warning("Could not import resource monitoring utilities. Some features may be limited.")
    RESOURCE_MONITORING_AVAILABLE = False

# Import resource metrics if available
try:
    from .resource_metrics import ResourceMetrics
except ImportError:
    # Define minimal implementation if import fails
    from dataclasses import dataclass, field
    from typing import Dict, Optional
    import time
    
    @dataclass
    class ResourceMetrics:
        """Minimal implementation of resource metrics"""
        timestamp: float = field(default_factory=time.time)
        cpu_usage: float = 0.0
        memory_usage: float = 0.0
        disk_usage: float = 0.0
        network_io: Dict[str, float] = field(default_factory=dict)
        gpu_usage: Optional[float] = None

__all__ = [
    'compute_accuracy',
    'compute_loss',
    'save_checkpoint',
    'load_checkpoint',
    'ResourceMetrics',
    'RESOURCE_MONITORING_AVAILABLE'
]
