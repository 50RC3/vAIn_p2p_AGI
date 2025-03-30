import asyncio
import logging
import time
import os
import psutil
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    gpu_memory: Optional[float] = None
    network_io: Optional[Dict[str, float]] = None

@dataclass
class NetworkHealthMetrics:
    peer_count: int = 0
    connection_success_rate: float = 0.0
    avg_latency: float = 0.0
    bandwidth_usage: float = 0.0
    overall_health: float = 0.0

class ResourceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self, check_interval: int = 30, interactive: bool = True):
        self.check_interval = check_interval
        self.interactive = interactive
        self.tracking_task = None
        self.is_tracking = False
        
    async def start_tracking(self) -> None:
        """Start tracking resource usage in background"""
        if self.tracking_task and not self.tracking_task.done():
            return
        
        self.is_tracking = True
        self.tracking_task = asyncio.create_task(self._track_resources())
        
    async def _track_resources(self) -> None:
        """Track resources in background"""
        while self.is_tracking:
            try:
                metrics = self.get_current_metrics()
                logger.debug(f"Resource metrics: CPU {metrics.cpu_usage:.1f}%, Memory {metrics.memory_usage:.1f}%")
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error tracking resources: {e}")
                await asyncio.sleep(self.check_interval * 2)  # Back off on error
                
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            # Try to get GPU info if available
            gpu_memory = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            except (ImportError, AttributeError):
                pass
                
            # Get network IO
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            }
            
            return ResourceMetrics(
                cpu_usage=cpu,
                memory_usage=memory,
                disk_usage=disk,
                gpu_memory=gpu_memory,
                network_io=network_io
            )
        except Exception as e:
            logger.error(f"Error getting resource metrics: {e}")
            return ResourceMetrics()
            
    async def check_resources_interactive(self) -> ResourceMetrics:
        """Check resource usage in interactive mode"""
        return self.get_current_metrics()
        
    def stop_tracking(self) -> None:
        """Stop tracking resources"""
        self.is_tracking = False
        if self.tracking_task and not self.tracking_task.done():
            self.tracking_task.cancel()

class NetworkMonitor:
    """Monitor network health and performance"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        self.monitoring_task = None
        self.stop_requested = False
        self.peer_metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        
    async def start_monitoring(self) -> None:
        """Start network monitoring in background"""
        if self.monitoring_task and not self.monitoring_task.done():
            return
            
        self.monitoring_task = asyncio.create_task(self._monitor_network())
        
    async def _monitor_network(self) -> None:
        """Monitor network in background"""
        while not self.stop_requested:
            try:
                metrics = await self.check_network_health()
                
                # Save metrics periodically
                current_timestamp = int(time.time())
                if current_timestamp % 300 == 0:  # Every 5 minutes
                    await self._save_metrics(metrics)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error monitoring network: {e}")
                await asyncio.sleep(60)  # Back off on error
                
    async def _save_metrics(self, metrics: NetworkHealthMetrics) -> None:
        """Save metrics to file"""
        try:
            import json
            
            timestamp = int(time.time())
            metrics_file = self.metrics_dir / f"network_metrics_{timestamp}.json"
            
            metrics_data = {
                "timestamp": timestamp,
                "peer_count": metrics.peer_count,
                "connection_success_rate": metrics.connection_success_rate,
                "avg_latency": metrics.avg_latency,
                "bandwidth_usage": metrics.bandwidth_usage,
                "overall_health": metrics.overall_health
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
                
    async def check_network_health(self) -> NetworkHealthMetrics:
        """Check network health"""
        # For the mock version, we'll return simulated metrics
        uptime = time.time() - self.start_time
        
        # No actual peers in this mock, but simulate some basic health indicators
        peer_count = 0
        success_rate = max(0.0, min(0.95, 0.5 + uptime / 3600))  # Improves over time up to 95%
        latency = max(50.0, 500.0 - uptime / 100)  # Decreases over time down to 50ms
        bandwidth = min(5.0, uptime / 1800)  # Increases over time up to 5 MB/s
        
        # Overall health is a weighted average of the above
        overall_health = 0.3 * success_rate + 0.3 * (1 - latency / 500) + 0.4 * (bandwidth / 5)
        
        return NetworkHealthMetrics(
            peer_count=peer_count,
            connection_success_rate=success_rate,
            avg_latency=latency,
            bandwidth_usage=bandwidth,
            overall_health=overall_health
        )
        
    def request_shutdown(self) -> None:
        """Request monitoring shutdown"""
        self.stop_requested = True
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()

def get_resource_metrics() -> ResourceMetrics:
    """Utility function to get current resource metrics"""
    try:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        return ResourceMetrics(
            cpu_usage=cpu,
            memory_usage=memory,
            disk_usage=disk
        )
    except Exception as e:
        logger.error(f"Error getting resource metrics: {e}")
        return ResourceMetrics()
