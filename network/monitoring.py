import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import time
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class NodeHealth:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]

class ResourceMonitor:
    def __init__(self, check_interval: int = 60, interactive: bool = True):
        self.check_interval = check_interval
        self.last_check = 0
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._last_metrics = None
        
    async def check_resources_interactive(self) -> Optional[NodeHealth]:
        """Interactive resource check with safety controls"""
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
                # Rate limiting
                if time.time() - self.last_check < self.check_interval:
                    if self._last_metrics:
                        return self._last_metrics
                    return None

                # Resource validation
                cpu_percent = psutil.cpu_percent()
                if cpu_percent > 90:
                    if self.interactive and not await self.session.confirm_with_timeout(
                        "High CPU usage detected. Continue monitoring?",
                        timeout=15
                    ):
                        return None

                # Get metrics with fallbacks
                try:
                    metrics = NodeHealth(
                        cpu_percent=cpu_percent,
                        memory_percent=psutil.virtual_memory().percent,
                        disk_percent=psutil.disk_usage('/').percent,
                        network_io=psutil.net_io_counters()._asdict()
                    )
                except Exception as e:
                    logger.error(f"Failed to get system metrics: {str(e)}")
                    if self._last_metrics:
                        logger.info("Using cached metrics")
                        return self._last_metrics
                    raise

                self.last_check = time.time()
                self._last_metrics = metrics
                return metrics

        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def check_resources(self) -> Optional[NodeHealth]:
        """Non-interactive fallback"""
        try:
            if time.time() - self.last_check < self.check_interval:
                return self._last_metrics
                
            metrics = NodeHealth(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                disk_percent=psutil.disk_usage('/').percent,
                network_io=psutil.net_io_counters()._asdict()
            )

            self.last_check = time.time()
            self._last_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return self._last_metrics

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'session') and self.session:
            self.session.cleanup()
            self.session = None
