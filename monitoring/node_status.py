import psutil
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class NodeMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]

class NodeMonitor:
    def __init__(self, sampling_interval: int = 60):
        self.sampling_interval = sampling_interval
        
    def get_metrics(self) -> NodeMetrics:
        return NodeMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io=self._get_network_stats()
        )
