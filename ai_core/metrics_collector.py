import asyncio
import logging
import time
import os
import json
import psutil
import threading
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
import torch
from datetime import datetime

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
        if self._is_running:
            return
            
        self._is_running = True
        self.last_collection = time.time()
        self.last_aggregation = time.time()
        self.last_cleanup = time.time()
        
        self.collection_task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Metrics collection started")
        
    async def stop(self) -> None:
        """Stop metrics collection"""
        if not self._is_running:
            return
            
        self._is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
            
        # Save final metrics
        await self._save_metrics()
        logger.info("Metrics collection stopped")
        
    async def _collect_metrics_loop(self) -> None:
        """Main metrics collection loop"""
        try:
            while self._is_running:
                await self._collect_metrics()
                
                # Check if it's time for aggregation
                if time.time() - self.last_aggregation >= self.config.aggregation_interval:
                    await self._aggregate_metrics()
                    self.last_aggregation = time.time()
                
                # Check if it's time for cleanup (once a day)
                if time.time() - self.last_cleanup >= 86400:  # 24 hours
                    await self._cleanup_old_metrics()
                    self.last_cleanup = time.time()
                
                # Save metrics periodically
                await self._save_metrics()
                
                # Sleep until next collection
                await asyncio.sleep(self.config.collection_interval)
                
        except asyncio.CancelledError:
            logger.info("Metrics collection cancelled")
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}", exc_info=True)
            
    async def _collect_metrics(self) -> None:
        """Collect current system metrics"""
        try:
            timestamp = time.time()
            
            # Basic system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Collect and store memory metrics
            await self._add_metric_point("memory_percent", memory.percent, timestamp)
            await self._add_metric_point("memory_available", memory.available, timestamp, 
                                        {"total": memory.total})
            
            # Collect and store CPU metrics
            await self._add_metric_point("cpu_percent", cpu_percent, timestamp)
            
            if self.config.process_metrics:
                # Process-specific metrics for this Python process
                process = psutil.Process(os.getpid())
                process_metrics = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "threads": len(process.threads()),
                    "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
                }
                
                for key, value in process_metrics.items():
                    await self._add_metric_point(f"process_{key}", value, timestamp)
            
            # GPU metrics if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        # Basic GPU metrics
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        gpu_percent = (allocated / total) * 100
                        
                        await self._add_metric_point(f"gpu{i}_memory_percent", gpu_percent, timestamp)
                        await self._add_metric_point(f"gpu{i}_memory_allocated", allocated, timestamp, 
                                                {"total": total})
                        
                        # Detailed GPU metrics if enabled
                        if self.config.detailed_gpu_metrics:
                            reserved = torch.cuda.memory_reserved(i)
                            cached = torch.cuda.memory_cached(i) if hasattr(torch.cuda, 'memory_cached') else 0
                            
                            await self._add_metric_point(f"gpu{i}_memory_reserved", reserved, timestamp)
                            await self._add_metric_point(f"gpu{i}_memory_cached", cached, timestamp)
                            
                    except Exception as e:
                        logger.warning(f"Error collecting GPU {i} metrics: {e}")
            
            # Network metrics if enabled
            if self.config.network_metrics:
                net_counters = psutil.net_io_counters()
                
                await self._add_metric_point("network_bytes_sent", net_counters.bytes_sent, timestamp)
                await self._add_metric_point("network_bytes_recv", net_counters.bytes_recv, timestamp)
                await self._add_metric_point("network_packets_sent", net_counters.packets_sent, timestamp)
                await self._add_metric_point("network_packets_recv", net_counters.packets_recv, timestamp)
                
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self._add_metric_point("disk_percent", disk.percent, timestamp)
            await self._add_metric_point("disk_free", disk.free, timestamp, 
                                         {"total": disk.total})
            
            # Update collection time
            self.last_collection = timestamp
            
            # Check for alerts
            await self._check_alerts()
            
            # Notify metric collection completed
            await self._notify_callbacks("metric_collected", {
                "timestamp": timestamp, 
                "metrics_count": sum(len(points) for points in self.metrics.values())
            })
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}", exc_info=True)
    
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
                
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}", exc_info=True)
    
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
                logger.warning(f"Metric alert: {alert['metric']} = {alert['value']:.1f}% (threshold: {alert['threshold']}%)")
                
        except Exception as e:
            logger.error(f"Error checking metric alerts: {e}", exc_info=True)
    
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
            with open(file_path, 'w') as f:
                json.dump(serialized, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)
    
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
            
            with open(file_path, 'w') as f:
                json.dump(self.aggregated_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving aggregated metrics: {e}", exc_info=True)
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period"""
        try:
            import datetime
            from datetime import timedelta
            import shutil
            
            # Calculate cutoff date
            now = datetime.datetime.now()
            cutoff_date = now - timedelta(days=self.config.retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            # Find directories older than cutoff
            for root, dirs, files in os.walk(self.config.storage_path):
                for dir_name in dirs:
                    try:
                        # Check if directory name is a date format (YYYY-MM-DD)
                        if len(dir_name) == 10 and dir_name[4] == '-' and dir_name[7] == '-':
                            if dir_name < cutoff_str:
                                dir_path = os.path.join(root, dir_name)
                                logger.info(f"Removing old metrics directory: {dir_path}")
                                shutil.rmtree(dir_path)
                    except Exception as e:
                        logger.warning(f"Error cleaning up directory {dir_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}", exc_info=True)
    
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
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
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
