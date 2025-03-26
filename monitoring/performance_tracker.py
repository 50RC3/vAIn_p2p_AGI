import time
import psutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from .alerts import AlertSystem, Alert

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, float]
    response_times: List[float]
    error_count: int
    active_connections: int

class PerformanceTracker:
    def __init__(self, alert_system: AlertSystem):
        self.alert_system = alert_system
        self.metrics_history: List[PerformanceMetrics] = []
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 80.0,
            'memory_critical': 90.0,
            'memory_warning': 80.0,
            'error_rate_critical': 0.1,
            'response_time_critical': 2.0  # seconds
        }
        
    async def track_performance(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            network_io=self._get_network_stats(),
            response_times=[],
            error_count=0,
            active_connections=len(psutil.net_connections())
        )
        
        self.metrics_history.append(metrics)
        await self._check_thresholds(metrics)
        return metrics

    def _get_network_stats(self) -> Dict[str, float]:
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }

    async def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        if metrics.cpu_usage >= self.thresholds['cpu_critical']:
            await self.alert_system.trigger_alert_interactive(Alert(
                severity='critical',
                message=f'Critical CPU usage: {metrics.cpu_usage:.1f}%',
                timestamp=datetime.now(),
                metadata={'metric': 'cpu_usage', 'value': metrics.cpu_usage}
            ))
        elif metrics.cpu_usage >= self.thresholds['cpu_warning']:
            await self.alert_system.trigger_alert_interactive(Alert(
                severity='warning',
                message=f'High CPU usage: {metrics.cpu_usage:.1f}%',
                timestamp=datetime.now(),
                metadata={'metric': 'cpu_usage', 'value': metrics.cpu_usage}
            ))

        if metrics.memory_usage >= self.thresholds['memory_critical']:
            await self.alert_system.trigger_alert_interactive(Alert(
                severity='critical', 
                message=f'Critical memory usage: {metrics.memory_usage:.1f}%',
                timestamp=datetime.now(),
                metadata={'metric': 'memory_usage', 'value': metrics.memory_usage}
            ))
