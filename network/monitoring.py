import psutil
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
import asyncio
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS
from collections import defaultdict
import statistics
import base64
import os

from monitoring.metrics_collector import MetricsCollector, SystemMetrics

logger = logging.getLogger(__name__)

@dataclass
class LoadBalanceMetrics:
    request_distribution: Dict[str, int] = field(default_factory=dict)
    node_capacity: float = 0.0
    current_load: float = 0.0
    load_variance: float = 0.0
    rebalance_count: int = 0
    transfer_latency: float = 0.0

@dataclass
class NodeHealth:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    load_metrics: Optional[LoadBalanceMetrics] = None

@dataclass 
class CacheStatus:
    last_update: float
    expiration: float
    hit_count: int = 0
    miss_count: int = 0

@dataclass
class NetworkHealth:
    overall_health: float
    network_load: float
    peer_count: int
    active_peers: int
    avg_latency: float
    bandwidth_usage: float

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
        self.metrics_collector = MetricsCollector(
            encryption_key=base64.urlsafe_b64encode(os.urandom(32)),
            metrics_dir="data/system_metrics"
        )
        self.alert_system = AlertSystem(interactive=interactive)
        self.performance_tracker = PerformanceTracker(self.alert_system)
        self.adaptive_thresholds = {
            'cpu': self._create_adaptive_threshold(cpu_threshold),
            'memory': self._create_adaptive_threshold(memory_threshold),
            'disk': self._create_adaptive_threshold(disk_threshold)
        }
        self.resource_history = []
        self.load_history = []
        self.last_rebalance = 0

    def _create_adaptive_threshold(self, base_threshold: float) -> Dict[str, float]:
        return {
            'base': base_threshold,
            'current': base_threshold,
            'min': max(base_threshold - 20, 50),
            'max': min(base_threshold + 10, 95)
        }

    def _adjust_thresholds(self, metrics: NodeHealth) -> None:
        """Dynamically adjust thresholds based on usage patterns"""
        self.resource_history.append(metrics)
        if len(self.resource_history) > 10:
            self.resource_history.pop(0)
            
        for metric, threshold in self.adaptive_thresholds.items():
            usage = getattr(metrics, f"{metric}_percent")
            avg_usage = sum(getattr(m, f"{metric}_percent") for m in self.resource_history) / len(self.resource_history)
            
            if avg_usage > threshold['current']:
                threshold['current'] = min(threshold['current'] * 1.1, threshold['max'])
            elif avg_usage < threshold['current'] * 0.7:
                threshold['current'] = max(threshold['current'] * 0.9, threshold['min'])

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
                    network_io=psutil.net_io_counters()._asdict(),
                    load_metrics=await self._get_load_metrics()
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

    async def _get_load_metrics(self) -> LoadBalanceMetrics:
        """Collect load balancing specific metrics"""
        try:
            current_time = time.time()
            
            # Calculate load metrics
            load_metrics = LoadBalanceMetrics(
                request_distribution=self._get_request_distribution(),
                node_capacity=self._calculate_node_capacity(),
                current_load=self._calculate_current_load(),
                load_variance=self._calculate_load_variance(),
                rebalance_count=len(self.load_history),
                transfer_latency=self._calculate_transfer_latency()
            )
            
            # Update history
            self.load_history.append((current_time, load_metrics))
            if len(self.load_history) > 100:  # Keep last 100 samples
                self.load_history.pop(0)
                
            return load_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect load metrics: {e}")
            return LoadBalanceMetrics()

    def _calculate_load_variance(self) -> float:
        """Calculate variance in load distribution"""
        if not self.load_history:
            return 0.0
        recent_loads = [m[1].current_load for m in self.load_history[-10:]]
        return float(np.var(recent_loads)) if recent_loads else 0.0

    def _calculate_transfer_latency(self) -> float:
        """Calculate average transfer latency for load balancing"""
        if not hasattr(self, '_transfer_times'):
            self._transfer_times = []
        return sum(self._transfer_times) / len(self._transfer_times) if self._transfer_times else 0.0

    async def _validate_resource_metrics(self, metrics: NodeHealth) -> tuple[bool, list[str]]:
        """Validate all resource metrics against thresholds"""
        warnings = []
        is_safe = True

        if metrics.cpu_percent > self.adaptive_thresholds['cpu']['current']:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}% > {self.adaptive_thresholds['cpu']['current']}%")
            is_safe = False

        if metrics.memory_percent > self.adaptive_thresholds['memory']['current']:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}% > {self.adaptive_thresholds['memory']['current']}%")
            is_safe = False

        if metrics.disk_percent > self.adaptive_thresholds['disk']['current']:
            warnings.append(f"Low disk space: {metrics.disk_percent:.1f}% > {self.adaptive_thresholds['disk']['current']}%")
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
                # Track performance
                await self.performance_tracker.track_performance()
                
                if not await self._wait_rate_limit():
                    return self._last_metrics

                # Resource validation with configurable threshold
                metrics = await self._get_metrics_with_retry()
                if metrics:
                    await self.metrics_collector.record_metric(
                        "system", 
                        "resources",
                        SystemMetrics(
                            cpu_usage=metrics.cpu_percent,
                            memory_usage=metrics.memory_percent,
                            network_io=self._get_network_stats(),
                            active_connections=len(psutil.net_connections())
                        )
                    )
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
                self._adjust_thresholds(metrics)
                return metrics

        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            await self.metrics_collector.record_metric("system", "errors", str(e))
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
            self._adjust_thresholds(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return self._last_metrics

    async def cleanup(self):
        """Enhanced cleanup with memory optimization"""
        try:
            if hasattr(self, 'session') and self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            
            self.resource_history.clear()
            self._last_metrics = None
            self._cache_status = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            gc.collect()
            
            logger.info("Resource monitor cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

from monitoring.health_monitor import SystemHealthMonitor

class NetworkMonitor:
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.load_history = []
        self.health_checks = {}
        self.peer_stats = defaultdict(dict)
        self.debug_metrics = []
        self._last_metric_time = 0
        self.metric_interval = 60  # Collect metrics every 60s
        self.health_monitor = SystemHealthMonitor()
        self._health_task = None
        
    async def start_monitoring(self):
        """Start network and health monitoring"""
        self._health_task = asyncio.create_task(
            self.health_monitor.monitor_health()
        )
        
    async def check_network_health(self) -> NetworkHealth:
        """Monitor overall network health with system metrics"""
        try:
            # Get system health status
            health = await self.health_monitor.system_health_check()
            if not health.status:
                debug_manager.track_error("Health check failed", {
                    "warnings": health.warnings,
                    "metrics": health.metrics
                })
            
            current_time = time.time()
            if current_time - self._last_metric_time >= self.metric_interval:
                self.debug_metrics.append(debug_manager.get_metrics())
                self._last_metric_time = current_time

            # Get node metrics
            node_health = await self.resource_monitor.check_resources_interactive()
            if not node_health:
                return None

            # Calculate network metrics
            peer_metrics = self.peer_stats.values()
            active_peers = sum(1 for p in peer_metrics if p.get('active', False))
            avg_latency = statistics.mean(p.get('latency', 0) for p in peer_metrics) 

            return NetworkHealth(
                overall_health=self._calculate_health_score(),
                network_load=sum(p.get('load', 0) for p in peer_metrics) / len(peer_metrics),
                peer_count=len(self.peer_stats),
                active_peers=active_peers,
                avg_latency=avg_latency,
                bandwidth_usage=health.metrics.get('network_throughput', 0)
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return None

    def _calculate_health_score(self) -> float:
        """Calculate health score with error tracking"""
        try:
            metrics = [
                self.resource_monitor.get_cache_stats().get('hit_ratio', 0),
                len(self.peer_stats) / 100, # Normalize to 0-1
                1 - (statistics.mean(p.get('error_rate', 0) for p in self.peer_stats.values()))
            ]
            return statistics.mean(metrics)
        except Exception as e:
            debug_manager.track_error(e)
            return 0.0

    async def update_peer_stats(self, peer_id: str, metrics: Dict):
        """Update peer performance metrics"""
        self.peer_stats[peer_id].update(metrics)
        self.peer_stats[peer_id]['last_update'] = time.time()

        # Cleanup old stats
        self._cleanup_old_stats()

    def cleanup(self):
        """Cleanup monitoring resources"""
        if self._health_task:
            self._health_task.cancel()
        if hasattr(self, 'session') and self.session:
            self.session.cleanup()
            self.session = None
