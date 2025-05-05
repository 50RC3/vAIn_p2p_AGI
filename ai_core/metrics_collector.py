import asyncio
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
import json
import psutil
import torch
import torch.cuda

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    collection_interval: int = 30  # seconds
    storage_path: str = "./logs/metrics"
    retention_days: int = 7
    detailed_gpu_metrics: bool = True
    network_metrics: bool = True
    process_metrics: bool = True
    aggregation_interval: int = 300  # 5 minutes
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "memory_percent": 85.0,
        "cpu_percent": 90.0,
        "gpu_memory_percent": 85.0,
        "disk_percent": 90.0
    })

@dataclass
class MetricPoint:
    """Single metric data point"""
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization"""
        # Ensure value is a float
        if not isinstance(self.value, (int, float)):
            try:
                self.value = float(self.value)
            except (ValueError, TypeError) as exc:
                raise TypeError(f"value must be numeric, got {type(self.value).__name__}") from exc
                
        # Ensure timestamp is a valid float
        if not isinstance(self.timestamp, (int, float)):
            try:
                self.timestamp = float(self.timestamp)
            except (ValueError, TypeError) as exc:
                raise TypeError(f"timestamp must be numeric, got {type(self.timestamp).__name__}") from exc
                
        # Ensure metadata is a dictionary
        if not isinstance(self.metadata, dict):
            self.metadata = {}
            
    def as_dict(self) -> Dict[str, Any]:
        """Return data point as a serializable dictionary"""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
        
    def __str__(self) -> str:
        """String representation of the metric point"""
        dt = datetime.fromtimestamp(self.timestamp)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        return f"MetricPoint(value={self.value:.2f}, time={time_str})"

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_collection: float = 0
        self.last_aggregation: float = 0
        self.last_cleanup: float = 0
        self.collection_task: Optional[asyncio.Task] = None
        self.callbacks: Dict[str, Set[Callable]] = {
            "alert": set(),
            "metric_collected": set(),
            "aggregation_completed": set()
        }
        self.lock = asyncio.Lock()
        self._is_running = False
        
        # Ensure metrics directory exists
        os.makedirs(self.config.storage_path, exist_ok=True)
        
    async def start(self) -> None:
        """Start metrics collection"""
        async with self.lock:
            if hasattr(self, '_is_running') and self._is_running:
                logger.warning("Metrics collection already running")
                return
                
            self._is_running = True
            self.last_collection = time.time()
            self.last_aggregation = time.time()
            self.last_cleanup = time.time()
            
            # Ensure storage directory exists
            os.makedirs(self.config.storage_path, exist_ok=True)
            
            self.collection_task = asyncio.create_task(self._collect_metrics_loop())
            logger.info("Metrics collection started")
    
    async def stop(self) -> None:
        """Stop metrics collection"""
        async with self.lock:
            if not hasattr(self, '_is_running') or not self._is_running:
                logger.warning("Metrics collection not running")
                return
                
            self._is_running = False
            
            if self.collection_task:
                self.collection_task.cancel()
                try:
                    await self.collection_task
                except asyncio.CancelledError:
                    logger.debug("Metrics collection task cancelled")
                
            logger.info("Metrics collection stopped")
        
    async def _collect_metrics_loop(self) -> None:
        """Background task to collect system metrics at regular intervals"""
        try:
            logger.info("Starting metrics collection loop")
            
            while self._is_running:
                start_time = time.time()
                
                try:
                    # Collect system metrics
                    await self._collect_system_metrics()
                    
                    # Perform aggregation if needed
                    if time.time() - self.last_aggregation >= self.config.aggregation_interval:
                        await self._aggregate_metrics()
                        self.last_aggregation = time.time()
                        
                    # Perform cleanup if needed (once per day)
                    if time.time() - self.last_cleanup >= 86400:
                        await self._cleanup_old_metrics()
                        self.last_cleanup = time.time()
                        
                except Exception as e:
                    logger.error("Error in metrics collection cycle: %s", e)
                
                # Calculate sleep time to maintain consistent collection interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.config.collection_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
        except Exception as e:
            logger.error("Metrics collection loop failed: %s", e)
    
    async def _collect_system_metrics(self) -> None:
        """Collect current system metrics"""
        try:
            # Collect CPU metrics
            cpu_metrics = await self._get_cpu_metrics()
            self._add_metric_point("cpu.percent", cpu_metrics.get("percent", 0.0))
            
            # Collect memory metrics
            memory_metrics = await self._get_memory_metrics()
            self._add_metric_point("memory.percent", memory_metrics.get("percent", 0.0))
            self._add_metric_point("memory.used", memory_metrics.get("used", 0.0))
            
            # Collect disk metrics
            disk_metrics = await self._get_disk_metrics()
            self._add_metric_point("disk.percent", disk_metrics.get("percent", 0.0))
            
            # Collect GPU metrics if enabled
            if self.config.detailed_gpu_metrics:
                gpu_metrics = await self._get_gpu_metrics()
                for idx, gpu_data in enumerate(gpu_metrics):
                    self._add_metric_point(
                        f"gpu.{idx}.memory_percent", 
                        gpu_data.get("memory_percent", 0.0)
                    )
                    self._add_metric_point(
                        f"gpu.{idx}.utilization", 
                        gpu_data.get("utilization", 0.0)
                    )
            
            # Collect network metrics if enabled
            if self.config.network_metrics:
                network_metrics = await self._get_network_metrics()
                self._add_metric_point("network.bytes_sent", network_metrics.get("bytes_sent", 0.0))
                self._add_metric_point("network.bytes_recv", network_metrics.get("bytes_recv", 0.0))
            
            # Collect process metrics if enabled
            if self.config.process_metrics:
                process_metrics = await self._get_process_metrics()
                self._add_metric_point("process.count", process_metrics.get("count", 0.0))
                self._add_metric_point(
                    "process.memory_percent", 
                    process_metrics.get("memory_percent", 0.0)
                )
            
            # Check for alerts
            await self._check_alerts()
            
            # Notify metric collected callbacks
            await self._notify_callbacks("metric_collected", {
                "timestamp": time.time(),
                "metrics_count": sum(len(points) for points in self.metrics.values())
            })
            
            self.last_collection = time.time()
            
        except Exception as e:
            logger.error("Error collecting system metrics: %s", e)
            raise
    
    async def _add_metric_point(self, name: str, value: float, timestamp: float, metadata: Dict[str, Any] = None) -> None:
        """Add a metric data point to the collection"""
        async with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
                
            # Add new point
            self.metrics[name].append(MetricPoint(
                value=value,
                timestamp=timestamp,
                metadata=metadata or {}
            ))
            
            # Limit number of points to avoid memory issues (keep last 1000 points)
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    async def _aggregate_metrics(self) -> None:
        """Create aggregated view of metrics"""
        try:
            async with self.lock:
                now = time.time()
                cutoff = now - self.config.aggregation_interval
                
                aggregated = {}
                
                for metric_name, points in self.metrics.items():
                    # Filter points from current aggregation interval
                    recent_points = [p for p in points if p.timestamp >= cutoff]
                    
                    if not recent_points:
                        continue
                        
                    values = [p.value for p in recent_points]
                    
                    # Calculate aggregation
                    aggregated[metric_name] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values),
                        "last": values[-1],
                        "start_time": cutoff,
                        "end_time": now
                    }
                
                self.aggregated_metrics = aggregated
                
                # Save aggregated metrics
                await self._save_aggregated_metrics()
                
                # Notify aggregation completed
                await self._notify_callbacks("aggregation_completed", {
                    "timestamp": now, 
                    "metrics_count": len(aggregated)
                })
                
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error("Error aggregating metrics: %s", e, exc_info=True)
        except (AttributeError, IndexError, OverflowError, MemoryError) as e:
            logger.error("Critical error during metrics aggregation: %s", e, exc_info=True)
    
    async def _check_alerts(self) -> None:
        """Check metrics against alert thresholds"""
        try:
            alerts = []
            
            # Check memory
            if "memory_percent" in self.metrics and self.metrics["memory_percent"]:
                memory_value = self.metrics["memory_percent"][-1].value
                threshold = self.config.alert_thresholds.get("memory_percent", 85.0)
                
                if memory_value > threshold:
                    alerts.append({
                        "metric": "memory_percent",
                        "value": memory_value,
                        "threshold": threshold,
                        "timestamp": time.time()
                    })
            
            # Check CPU
            if "cpu_percent" in self.metrics and self.metrics["cpu_percent"]:
                cpu_value = self.metrics["cpu_percent"][-1].value
                threshold = self.config.alert_thresholds.get("cpu_percent", 90.0)
                
                if cpu_value > threshold:
                    alerts.append({
                        "metric": "cpu_percent",
                        "value": cpu_value,
                        "threshold": threshold,
                        "timestamp": time.time()
                    })
            
            # Check GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    metric_name = f"gpu{i}_memory_percent"
                    if metric_name in self.metrics and self.metrics[metric_name]:
                        gpu_value = self.metrics[metric_name][-1].value
                        threshold = self.config.alert_thresholds.get("gpu_memory_percent", 85.0)
                        
                        if gpu_value > threshold:
                            alerts.append({
                                "metric": metric_name,
                                "value": gpu_value,
                                "threshold": threshold,
                                "timestamp": time.time(),
                                "gpu_id": i
                            })
            
            # Check disk space
            if "disk_percent" in self.metrics and self.metrics["disk_percent"]:
                disk_value = self.metrics["disk_percent"][-1].value
                threshold = self.config.alert_thresholds.get("disk_percent", 90.0)
                
                if disk_value > threshold:
                    alerts.append({
                        "metric": "disk_percent",
                        "value": disk_value,
                        "threshold": threshold,
                        "timestamp": time.time()
                    })
            
            # Send alerts
            for alert in alerts:
                await self._notify_callbacks("alert", alert)
                logger.warning("Metric alert: %s = %.1f%% (threshold: %s%%)", alert['metric'], alert['value'], alert['threshold'])
                
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error("Error checking metric alerts: %s", e, exc_info=True)
        except (AttributeError, IndexError, LookupError, ArithmeticError) as e:
            logger.error("Critical error during alert checking: %s", e, exc_info=True)
    
    async def _save_metrics(self) -> None:
        """Save current metrics to storage"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            hour_str = now.strftime("%H")
            
            # Create directory structure for metrics
            metrics_dir = os.path.join(self.config.storage_path, date_str, hour_str)
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Save metrics to file
            file_path = os.path.join(metrics_dir, f"metrics_{now.strftime('%Y%m%d_%H%M%S')}.json")
            
            # Convert metrics to serializable format
            serialized = self._serialize_metrics()
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, indent=2)
                
        except (IOError, OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error("Error saving metrics: %s", e, exc_info=True)
    
    async def _save_aggregated_metrics(self) -> None:
        """Save aggregated metrics to storage"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            
            # Create directory for aggregated metrics
            aggregated_dir = os.path.join(self.config.storage_path, 'aggregated', date_str)
            os.makedirs(aggregated_dir, exist_ok=True)
            
            # Save aggregated metrics to file
            file_path = os.path.join(aggregated_dir, f"aggregated_{now.strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.aggregated_metrics, f, indent=2)
                
        except (IOError, OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error("Error saving aggregated metrics: %s", e, exc_info=True)
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period"""
        try:
            from datetime import timedelta
            import shutil
            
            # Calculate cutoff date
            now = datetime.now()
            cutoff_date = now - timedelta(days=self.config.retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            # Find directories older than cutoff
            for root, dirs, _ in os.walk(self.config.storage_path):
                for dir_name in dirs:
                    try:
                        # Check if directory name is a date format (YYYY-MM-DD)
                        if len(dir_name) == 10 and dir_name[4] == '-' and dir_name[7] == '-':
                            if dir_name < cutoff_str:
                                dir_path = os.path.join(root, dir_name)
                                logger.info("Removing old metrics directory: %s", dir_path)
                                shutil.rmtree(dir_path)
                    except (OSError, IOError, PermissionError) as e:
                        logger.warning("Error cleaning up directory %s: %s", dir_name, e)
                        
        except (OSError, IOError, PermissionError, ValueError, TypeError) as e:
            logger.error("Error cleaning up old metrics: %s", e, exc_info=True)
        except (AttributeError, LookupError, ZeroDivisionError, MemoryError) as e:
            logger.error("Unexpected error during metrics cleanup: %s", e, exc_info=True)
    
    def _serialize_metrics(self) -> Dict[str, Any]:
        """Convert metrics to a serializable format"""
        result = {}
        
        for metric_name, points in self.metrics.items():
            result[metric_name] = [
                {
                    "value": point.value,
                    "timestamp": point.timestamp,
                    "metadata": point.metadata
                }
                for point in points
            ]
            
        return result
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for metrics events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """Unregister a callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify all registered callbacks for an event"""
        if event_type not in self.callbacks:
            return
            
        for callback in list(self.callbacks[event_type]):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as e:
                logger.error("Error in metrics callback: %s", e)
    
    async def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent value for each metric"""
        result = {}
        
        async with self.lock:
            for name, points in self.metrics.items():
                if points:
                    result[name] = points[-1].value
                    
        return result
    
    async def get_metric_history(self, metric_name: str, 
                               start_time: Optional[float] = None, 
                               end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        if metric_name not in self.metrics:
            return []
            
        now = time.time()
        start_time = start_time or (now - 3600)  # Default: last hour
        end_time = end_time or now
        
        points = []
        
        async with self.lock:
            for point in self.metrics[metric_name]:
                if start_time <= point.timestamp <= end_time:
                    points.append({
                        "value": point.value,
                        "timestamp": point.timestamp,
                        "metadata": point.metadata
                    })
                    
        return points
