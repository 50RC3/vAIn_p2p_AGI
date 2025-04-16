"""
Resource Monitor - Monitors system resources and reports to the system coordinator
"""
import asyncio
import logging
import psutil
import time
from typing import Dict, Any, Optional, Callable
import os
from pathlib import Path

from core.system_coordinator import get_coordinator

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitors system resources and reports to the coordinator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize resource monitor.
        
        Args:
            config: Configuration dictionary for the monitor
        """
        self.config = config or {
            "check_interval": 30,  # seconds
            "cpu_warning_threshold": 80.0,  # percentage
            "memory_warning_threshold": 75.0,  # percentage
            "disk_warning_threshold": 90.0,  # percentage
            "log_directory": "logs/resources",
            "history_size": 100  # number of data points to keep
        }
        
        # Set up monitoring state
        self.running = False
        self.monitor_task = None
        self.history = []
        self._lock = asyncio.Lock()
        self.callbacks = {}
        
        # Connect to system coordinator if available
        try:
            self.coordinator = get_coordinator()
        except Exception as e:
            logger.warning(f"Could not connect to system coordinator: {e}")
            self.coordinator = None
            
        # Create log directory
        os.makedirs(self.config["log_directory"], exist_ok=True)
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")
        
        # Register with coordinator if available
        if self.coordinator:
            self.coordinator.register_component("resource_monitor", self)
        
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Resource monitoring stopped")
        
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for resource events"""
        self.callbacks[event_type] = callback
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.running:
                # Get current resource stats
                stats = self._get_resource_stats()
                
                # Add to history
                async with self._lock:
                    self.history.append(stats)
                    # Trim history if needed
                    if len(self.history) > self.config["history_size"]:
                        self.history = self.history[-self.config["history_size"]:]
                
                # Check thresholds
                await self._check_thresholds(stats)
                
                # Log to file periodically
                if len(self.history) % 10 == 0:
                    self._log_to_file()
                    
                # Wait for next check
                await asyncio.sleep(self.config["check_interval"])
                
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
            
    def _get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Include GPU stats if available
        gpu_stats = self._get_gpu_stats()
        
        stats = {
            "timestamp": time.time(),
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent
            }
        }
        
        if gpu_stats:
            stats["gpu"] = gpu_stats
            
        return stats
    
    def _get_gpu_stats(self) -> Optional[Dict[str, Any]]:
        """Get GPU statistics if available"""
        try:
            # Try PyTorch
            import torch
            if torch.cuda.is_available():
                return {
                    "count": torch.cuda.device_count(),
                    "devices": [{
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "max_memory": torch.cuda.get_device_properties(i).total_memory
                    } for i in range(torch.cuda.device_count())]
                }
        except:
            pass
            
        try:
            # Try with nvidia-smi
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                           stdout=subprocess.PIPE, text=True).stdout.strip()
            if result:
                lines = result.split('\n')
                devices = []
                for line in lines:
                    name, used, total = line.split(',')
                    devices.append({
                        "name": name.strip(),
                        "memory_used": int(used.strip()) * 1024 * 1024,  # Convert to bytes
                        "memory_total": int(total.strip()) * 1024 * 1024  # Convert to bytes
                    })
                return {
                    "count": len(devices),
                    "devices": devices
                }
        except:
            pass
            
        return None
        
    async def _check_thresholds(self, stats: Dict[str, Any]):
        """Check if any resource is above threshold"""
        warnings = []
        
        # Check CPU
        if stats["cpu"]["percent"] > self.config["cpu_warning_threshold"]:
            warnings.append(f"CPU usage high: {stats['cpu']['percent']}%")
            
        # Check memory
        if stats["memory"]["percent"] > self.config["memory_warning_threshold"]:
            warnings.append(f"Memory usage high: {stats['memory']['percent']}%")
            
        # Check disk
        if stats["disk"]["percent"] > self.config["disk_warning_threshold"]:
            warnings.append(f"Disk usage high: {stats['disk']['percent']}%")
            
        # Check GPU if available
        if "gpu" in stats:
            for i, device in enumerate(stats["gpu"]["devices"]):
                if "memory_allocated" in device and "max_memory" in device:
                    usage_percent = device["memory_allocated"] / device["max_memory"] * 100
                    if usage_percent > self.config["memory_warning_threshold"]:
                        warnings.append(f"GPU {i} memory usage high: {usage_percent:.1f}%")
                        
        # Log warnings
        if warnings:
            for warning in warnings:
                logger.warning(warning)
            
            # Dispatch events
            if self.coordinator:
                self.coordinator.dispatch_event("resource_warning", {
                    "warnings": warnings,
                    "stats": stats
                })
                
            # Call callbacks
            if "warning" in self.callbacks:
                self.callbacks["warning"](warnings)
                
            # Check for critical thresholds
            if stats["memory"]["percent"] > 90:
                if self.coordinator:
                    self.coordinator.dispatch_event("critical_memory", {
                        "usage": stats["memory"]["percent"] / 100,
                        "available": stats["memory"]["available"],
                        "source": "resource_monitor"
                    })
                    
    def _log_to_file(self):
        """Log resource stats to file"""
        try:
            if not self.history:
                return
                
            log_path = Path(self.config["log_directory"]) / f"resources_{int(time.time())}.csv"
            with open(log_path, "w") as f:
                # Write header
                f.write("timestamp,cpu_percent,memory_percent,disk_percent\n")
                
                # Write data
                for stats in self.history:
                    f.write(f"{stats['timestamp']},{stats['cpu']['percent']},{stats['memory']['percent']},{stats['disk']['percent']}\n")
                    
        except Exception as e:
            logger.error(f"Error logging resource stats to file: {e}")
            
    async def get_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        if not self.history:
            current_stats = self._get_resource_stats()
        else:
            current_stats = self.history[-1]
            
        return {
            "running": self.running,
            "history_size": len(self.history),
            "current_stats": current_stats
        }
        
    async def get_resource_trends(self, minutes: int = 10) -> Dict[str, Any]:
        """Get resource usage trends over the specified time period"""
        async with self._lock:
            if not self.history or len(self.history) < 2:
                return {"error": "Not enough data points for trend analysis"}
                
            now = time.time()
            cutoff = now - (minutes * 60)
            
            # Filter history to the requested time window
            relevant_history = [stats for stats in self.history if stats["timestamp"] >= cutoff]
            
            if len(relevant_history) < 2:
                return {"error": "Not enough data points in the requested time window"}
                
            # Calculate trends
            first = relevant_history[0]
            last = relevant_history[-1]
            
            cpu_change = last["cpu"]["percent"] - first["cpu"]["percent"]
            memory_change = last["memory"]["percent"] - first["memory"]["percent"]
            disk_change = last["disk"]["percent"] - first["disk"]["percent"]
            
            # Calculate rates per minute
            time_span_minutes = (last["timestamp"] - first["timestamp"]) / 60
            if time_span_minutes > 0:
                cpu_rate = cpu_change / time_span_minutes
                memory_rate = memory_change / time_span_minutes
                disk_rate = disk_change / time_span_minutes
            else:
                cpu_rate = memory_rate = disk_rate = 0
                
            return {
                "time_span_minutes": time_span_minutes,
                "cpu": {
                    "change": cpu_change,
                    "rate": cpu_rate,
                    "trend": "increasing" if cpu_rate > 0.5 else "decreasing" if cpu_rate < -0.5 else "stable"
                },
                "memory": {
                    "change": memory_change,
                    "rate": memory_rate,
                    "trend": "increasing" if memory_rate > 0.5 else "decreasing" if memory_rate < -0.5 else "stable"
                },
                "disk": {
                    "change": disk_change,
                    "rate": disk_rate,
                    "trend": "increasing" if disk_rate > 0.5 else "decreasing" if disk_rate < -0.5 else "stable"
                },
                "current": {
                    "cpu": last["cpu"]["percent"],
                    "memory": last["memory"]["percent"],
                    "disk": last["disk"]["percent"]
                }
            }
            
    async def shutdown(self):
        """Shutdown the resource monitor"""
        await self.stop_monitoring()
