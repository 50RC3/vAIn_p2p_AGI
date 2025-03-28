from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any
import psutil
import torch
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Tracks system resource utilization metrics."""
    timestamp: float = field(default_factory=time.time)
    value: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    gpu_usage: Optional[float] = None
    latency_ms: float = 0.0
    error_count: int = 0

    @classmethod
    def get_current(cls) -> 'ResourceMetrics':
        metrics = cls(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
        )
        if torch.cuda.is_available():
            metrics.gpu_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return metrics

    def update_metrics(self) -> None:
        """Update all resource metrics."""
        try:
            self.timestamp = time.time()
            self.cpu_usage = psutil.cpu_percent()
            self.memory_usage = psutil.virtual_memory().percent
            self.disk_usage = psutil.disk_usage('/').percent
            
            net_io = psutil.net_io_counters()
            self.network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }

            if torch.cuda.is_available():
                self.gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0

        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
            self.error_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'gpu_usage': self.gpu_usage,
            'latency_ms': self.latency_ms,
            'error_count': self.error_count
        }

    def update_value(self, value: float) -> None:
        """Update the value metric."""
        self.value = value
        self.timestamp = time.time()

    def get_load_factor(self) -> float:
        """Calculate overall system load factor."""
        metrics = [self.cpu_usage, self.memory_usage, self.disk_usage]
        if self.gpu_usage is not None:
            metrics.append(self.gpu_usage)
        return sum(metrics) / len(metrics) / 100.0

    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """Check if system is overloaded."""
        return self.get_load_factor() > threshold
