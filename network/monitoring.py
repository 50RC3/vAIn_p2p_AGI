import psutil
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time
import asyncio
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class NodeHealth:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]

@dataclass 
class CacheStatus:
    last_update: float
    expiration: float
    hit_count: int = 0
    miss_count: int = 0

class ResourceMonitor:
    def __init__(self, check_interval: int = 60, interactive: bool = True,
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 90.0,
                 disk_threshold: float = 95.0,
                 max_retries: int = 3,
                 initial_retry_delay: float = 1.0,
                 base_cache_duration: int = 300,  # 5 minutes base cache duration
                 min_cache_duration: int = 60,    # 1 minute minimum
                 max_cache_duration: int = 1800): # 30 minutes maximum
        self.check_interval = check_interval
        self.interactive = interactive
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.last_check = 0
        self.consecutive_failures = 0
        self._interrupt_requested = False
        self._last_metrics = None
        self._bypass_rate_limit = False
        self.base_cache_duration = base_cache_duration
        self.min_cache_duration = min_cache_duration
        self.max_cache_duration = max_cache_duration
        self._cache_status = None

    def _calculate_cache_duration(self, metrics: NodeHealth) -> int:
        """Calculate dynamic cache duration based on system load"""
        # Reduce cache duration under high load
        load_factor = max(
            metrics.cpu_percent / 100,
            metrics.memory_percent / 100,
            metrics.disk_percent / 100
        )
        
        # Exponentially reduce cache duration as load increases
        dynamic_duration = int(self.base_cache_duration * (1 - load_factor**2))
        return max(min(dynamic_duration, self.max_cache_duration), self.min_cache_duration)

    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid"""
        if not self._last_metrics or not self._cache_status:
            return False
            
        current_time = time.time()
        if current_time > self._cache_status.expiration:
            return False
            
        return True

    def _update_cache(self, metrics: NodeHealth) -> None:
        """Update cache with new metrics and status"""
        self._last_metrics = metrics
        cache_duration = self._calculate_cache_duration(metrics)
        current_time = time.time()
        
        if self._cache_status:
            self._cache_status.last_update = current_time
            self._cache_status.expiration = current_time + cache_duration
        else:
            self._cache_status = CacheStatus(
                last_update=current_time,
                expiration=current_time + cache_duration
            )

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        if not self._cache_status:
            return {}
            
        total_requests = self._cache_status.hit_count + self._cache_status.miss_count
        hit_ratio = (self._cache_status.hit_count / total_requests 
                    if total_requests > 0 else 0)
                    
        return {
            'hit_ratio': hit_ratio,
            'total_requests': total_requests,
            'current_duration': self._cache_status.expiration - self._cache_status.last_update
        }

    async def _wait_rate_limit(self) -> bool:
        """Wait for rate limit or return False if should skip check"""
        if self._bypass_rate_limit:
            self._bypass_rate_limit = False
            return True

        if self._is_cache_valid():
            self._cache_status.hit_count += 1
            return False
            
        self._cache_status.miss_count += 1 if self._cache_status else 0
        time_since_check = time.time() - self.last_check
        if time_since_check < self.check_interval:
            if self._last_metrics:
                return False
            await asyncio.sleep(self.check_interval - time_since_check)
        return True

    def force_check(self) -> None:
        """Force next check to bypass rate limiting"""
        self._bypass_rate_limit = True

    async def _get_metrics_with_retry(self) -> Optional[NodeHealth]:
        """Get metrics with exponential backoff retry"""
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                metrics = NodeHealth(
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=psutil.virtual_memory().percent,
                    disk_percent=psutil.disk_usage('/').percent,
                    network_io=psutil.net_io_counters()._asdict()
                )
                self.consecutive_failures = 0
                return metrics
            except Exception as e:
                retry_count += 1
                self.consecutive_failures += 1
                if retry_count <= self.max_retries:
                    delay = self.initial_retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Metrics collection failed, retrying in {delay:.1f}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to collect metrics after {retry_count} attempts: {str(e)}")
        return None

    async def _validate_resource_metrics(self, metrics: NodeHealth) -> tuple[bool, list[str]]:
        """Validate all resource metrics against thresholds"""
        warnings = []
        is_safe = True

        if metrics.cpu_percent > self.cpu_threshold:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}% > {self.cpu_threshold}%")
            is_safe = False

        if metrics.memory_percent > self.memory_threshold:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}% > {self.memory_threshold}%")
            is_safe = False

        if metrics.disk_percent > self.disk_threshold:
            warnings.append(f"Low disk space: {metrics.disk_percent:.1f}% > {self.disk_threshold}%")
            is_safe = False

        return is_safe, warnings

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
                if not await self._wait_rate_limit():
                    return self._last_metrics

                # Resource validation with configurable threshold
                metrics = await self._get_metrics_with_retry()
                if not metrics:
                    return self._last_metrics

                is_safe, warnings = await self._validate_resource_metrics(metrics)
                if not is_safe and self.interactive:
                    warning_msg = "\n- ".join(["Resource warnings:"] + warnings)
                    if not await self.session.confirm_with_timeout(
                        f"{warning_msg}\nContinue monitoring? (y/n): ",
                        timeout=15
                    ):
                        return None

                self.last_check = time.time()
                self._update_cache(metrics)
                return metrics

        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def check_resources(self) -> Optional[NodeHealth]:
        """Non-interactive fallback with retry mechanism"""
        try:
            if time.time() - self.last_check < self.check_interval and not self._bypass_rate_limit:
                return self._last_metrics
                
            self._bypass_rate_limit = False
            metrics = asyncio.run(self._get_metrics_with_retry())
            if not metrics:
                return self._last_metrics

            # Log warnings for non-interactive mode
            is_safe, warnings = asyncio.run(self._validate_resource_metrics(metrics))
            if not is_safe:
                for warning in warnings:
                    logger.warning(warning)

            self.last_check = time.time()
            self._update_cache(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return self._last_metrics

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'session') and self.session:
            self.session.cleanup()
            self.session = None
