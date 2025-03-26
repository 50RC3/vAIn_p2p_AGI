import psutil
import time
from dataclasses import dataclass
from typing import Dict, Optional
import logging
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class NodeMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]

class NodeMonitor:
    def __init__(self, sampling_interval: int = 60, interactive: bool = True):
        self.sampling_interval = sampling_interval
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self.alert_system = AlertSystem(interactive=interactive)
        self.performance_tracker = PerformanceTracker(self.alert_system)

    async def get_metrics_interactive(self) -> Optional[NodeMetrics]:
        """Get node metrics with interactive monitoring and safety checks"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["resource_check"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Track performance before getting metrics
                await self.performance_tracker.track_performance()
                
                # Resource validation
                if psutil.cpu_percent() > 90:
                    if not await self.session.confirm_with_timeout(
                        "High CPU usage detected. Continue monitoring?",
                        timeout=15
                    ):
                        return None

                metrics = self.get_metrics()
                
                # Cache metrics for recovery
                await self._save_progress(metrics.__dict__)
                
                return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def get_metrics(self) -> NodeMetrics:
        return NodeMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io=self._get_network_stats()
        )

    def _get_network_stats(self) -> Dict[str, float]:
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }

    async def _save_progress(self, metrics: Dict) -> None:
        """Save metrics for recovery"""
        if self.session:
            await self.session.save_progress(metrics)

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'session') and self.session:
            self.session.cleanup()
        if hasattr(self, 'alert_system'):
            self.alert_system.cleanup()
