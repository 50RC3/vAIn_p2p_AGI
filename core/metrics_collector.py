import logging
import time
import statistics
from typing import Dict, List, Optional
import psutil
from dataclasses import dataclass, field
from threading import Lock
import torch

logger = logging.getLogger(__name__)

@dataclass
class MetricsSnapshot:
    timestamp: float
    node_count: int
    active_validations: int
    memory_usage: float
    cpu_usage: float
    network_latency: List[float]
    validation_times: List[float]
    gpu_memory: Optional[float] = None

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'node_count': 0,
            'active_validations': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'network_latency': [],
            'validation_times': []
        }
        self._lock = Lock()
        self.history: List[MetricsSnapshot] = []
        self.max_history_size = 1000
        self.sampling_interval = 60  # seconds

    def update_node_metrics(self, count: int) -> None:
        with self._lock:
            self.metrics['node_count'] = count

    def record_validation(self, duration: float, success: bool) -> None:
        with self._lock:
            self.metrics['active_validations'] += 1
            self.metrics['validation_times'].append(duration)
            # Keep only last 1000 validation times
            if len(self.metrics['validation_times']) > 1000:
                self.metrics['validation_times'] = self.metrics['validation_times'][-1000:]

    def record_latency(self, latency: float) -> None:
        with self._lock:
            self.metrics['network_latency'].append(latency)
            if len(self.metrics['network_latency']) > 1000:
                self.metrics['network_latency'] = self.metrics['network_latency'][-1000:]

    def _update_resource_metrics(self) -> None:
        self.metrics['cpu_usage'] = psutil.cpu_percent()
        self.metrics['memory_usage'] = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            self.metrics['gpu_memory'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0

    def get_current_metrics(self) -> Dict:
        with self._lock:
            metrics = self.metrics.copy()
            metrics.update({
                'avg_latency': statistics.mean(self.metrics['network_latency']) if self.metrics['network_latency'] else 0,
                'avg_validation_time': statistics.mean(self.metrics['validation_times']) if self.metrics['validation_times'] else 0,
                'timestamp': time.time()
            })
            return metrics

    def create_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            self._update_resource_metrics()
            snapshot = MetricsSnapshot(
                timestamp=time.time(),
                **self.metrics
            )
            self.history.append(snapshot)
            if len(self.history) > self.max_history_size:
                self.history = self.history[-self.max_history_size:]
            return snapshot

    def reset_validation_metrics(self) -> None:
        with self._lock:
            self.metrics['active_validations'] = 0
            self.metrics['validation_times'] = []
            self.metrics['network_latency'] = []
