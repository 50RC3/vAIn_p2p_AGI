import psutil
import logging
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    status: bool
    details: Dict[str, bool]
    metrics: Dict[str, float]
    warnings: list[str]

class SystemHealthMonitor:
    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or {
            'memory': 90.0,  # 90% usage
            'cpu': 90.0,     # 90% usage
            'storage': 95.0, # 95% usage
            'network': 1000  # Minimum bytes/sec
        }
        
    async def system_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        warnings = []
        metrics = {}
        
        # Get system metrics
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        metrics['cpu_usage'] = psutil.cpu_percent()
        metrics['storage_usage'] = psutil.disk_usage('/').percent
        
        net_io = psutil.net_io_counters()
        metrics['network_throughput'] = net_io.bytes_sent + net_io.bytes_recv
        
        if torch.cuda.is_available():
            metrics['gpu_memory'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100

        # Perform checks
        checks = {
            'memory': memory.percent < self.thresholds['memory'],
            'cpu': metrics['cpu_usage'] < self.thresholds['cpu'],
            'storage': metrics['storage_usage'] < self.thresholds['storage'],
            'network': metrics['network_throughput'] > self.thresholds['network']
        }
        
        # Collect warnings
        if not checks['memory']:
            warnings.append(f"High memory usage: {memory.percent:.1f}%")
        if not checks['cpu']:
            warnings.append(f"High CPU usage: {metrics['cpu_usage']:.1f}%")
        if not checks['storage']:
            warnings.append(f"Low storage space: {metrics['storage_usage']:.1f}%")
        
        return SystemHealth(
            status=all(checks.values()),
            details=checks,
            metrics=metrics,
            warnings=warnings
        )

    async def monitor_health(self, interval: int = 60):
        """Continuous health monitoring"""
        while True:
            try:
                health = await self.system_health_check()
                if not health.status:
                    logger.warning("Health check failed:")
                    for warning in health.warnings:
                        logger.warning(warning)
                        
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(INTERACTION_TIMEOUTS["emergency"])
