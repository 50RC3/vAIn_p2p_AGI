import logging
import psutil
import torch
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float]
    network_io: Dict[str, int]
    error_count: int
    latency_ms: float
    timestamp: float

class SystemMonitor:
    def __init__(self, 
                 check_interval: int = 60,
                 metrics_history_size: int = 1000,
                 log_dir: Optional[str] = None):
        self.check_interval = check_interval
        self.metrics_history = []
        self.max_history = metrics_history_size
        self.log_dir = Path(log_dir) if log_dir else None
        self._setup_logging()
        self.error_counts: Dict[str, int] = {}
        self._monitoring = False

    def _setup_logging(self):
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True)
            fh = logging.FileHandler(self.log_dir / "system_monitor.log")
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(fh)

    async def start_monitoring(self):
        """Start continuous system monitoring"""
        self._monitoring = True
        while self._monitoring:
            try:
                metrics = await self.get_metrics()
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                    
                if self._check_critical_levels(metrics):
                    logger.warning("Critical resource usage detected")
                    
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False

    async def get_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            gpu_usage=gpu_usage,
            network_io=dict(psutil.net_io_counters()._asdict()),
            error_count=sum(self.error_counts.values()),
            latency_ms=0,
            timestamp=time.time()
        )

    def _check_critical_levels(self, metrics: SystemMetrics) -> bool:
        """Check for critical resource usage"""
        return any([
            metrics.cpu_usage > 90,
            metrics.memory_usage > 90,
            metrics.disk_usage > 95,
            metrics.gpu_usage is not None and metrics.gpu_usage > 0.95
        ])

    def track_error(self, error: Exception, context: Optional[Dict] = None):
        """Track error occurrence"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        logger.error(f"Error occurred: {str(error)}", 
                    exc_info=True,
                    extra={"context": context})

    def get_error_summary(self) -> Dict[str, int]:
        """Get error count summary"""
        return dict(self.error_counts)

    async def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
