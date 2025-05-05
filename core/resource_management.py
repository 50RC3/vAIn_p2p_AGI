import logging
import time
import platform
import psutil
import threading
import os
import gc
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages system resources for the vAIn P2P AGI system.
    Monitors CPU, memory, disk, and network usage.
    Provides resource allocation and throttling capabilities.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0,
                 critical_memory_threshold: float = 200.0):  # Added this parameter
        """
        Initialize the resource manager.
        
        Args:
            monitoring_interval: Interval in seconds for resource monitoring
            cpu_threshold: CPU usage percentage threshold for alerts
            memory_threshold: Memory usage percentage threshold for alerts
            disk_threshold: Disk usage percentage threshold for alerts
            critical_memory_threshold: Memory threshold in MB for critical alerts
        """
        self.monitoring_interval = monitoring_interval
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.critical_memory_threshold = critical_memory_threshold  # Critical threshold in MB
        
        self._shutdown_requested = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._module_allocations: Dict[str, Dict[str, float]] = {}
        self._resource_lock = threading.RLock()
        self._last_measurements: Dict[str, Any] = {}
        self._subscribers = []  # Added subscriber list
        
        # System info
        try:
            total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB
        except (ValueError, TypeError):
            total_memory = 0
            
        self.system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory": total_memory,
        }
        
        logger.info("Resource Manager initialized on %s with %s CPUs and %.2f GB RAM",
                   self.system_info['platform'],
                   self.system_info['cpu_count'],
                   self.system_info['total_memory'] / 1024)  # MB to GB
    
    def start_monitoring(self) -> None:
        """Start the resource monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring is already running")
            return
            
        self._shutdown_requested = False
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop the resource monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring is not running")
            return
            
        self._shutdown_requested = True
        self._monitoring_thread.join(timeout=10)
        if self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring thread did not terminate cleanly")
        else:
            logger.info("Resource monitoring stopped")
            
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that periodically checks resource usage."""
        while not self._shutdown_requested:
            try:
                self._check_resources()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error("Error in resource monitoring: %s", e)
                time.sleep(max(1.0, self.monitoring_interval / 2))
                
    def _check_resources(self) -> None:
        """Check current resource usage and store measurements."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network IO
            net_io = psutil.net_io_counters()
            
            with self._resource_lock:
                # Store measurements
                timestamp = time.time()
                self._last_measurements = {
                    "timestamp": timestamp,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_used": memory.used,
                    "memory_total": memory.total,
                    "disk_percent": disk_percent,
                    "disk_used": disk.used,
                    "disk_total": disk.total,
                    "net_bytes_sent": net_io.bytes_sent,
                    "net_bytes_recv": net_io.bytes_recv,
                }
                
                # Check thresholds and log warnings
                if cpu_percent > self.cpu_threshold:
                    logger.warning("CPU usage is high: %s%%", cpu_percent)
                
                if memory_percent > self.memory_threshold:
                    logger.warning("Memory usage is high: %s%%", memory_percent)
                
                if disk_percent > self.disk_threshold:
                    logger.warning("Disk usage is high: %s%%", disk_percent)
                    
        except Exception as e:
            logger.error("Error checking resources: %s", e)
            
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get the current resource usage.
        
        Returns:
            Dict containing resource usage measurements
        """
        with self._resource_lock:
            return dict(self._last_measurements)
            
    def allocate_resources(self, module_id: str, cpu_percent: float = 0, 
                          memory_mb: float = 0) -> bool:
        """
        Allocate resources for a module.
        
        Args:
            module_id: ID of the module requesting resources
            cpu_percent: Percentage of CPU to allocate (0-100)
            memory_mb: Memory in MB to allocate
            
        Returns:
            True if resources were allocated, False otherwise
        """
        with self._resource_lock:
            # Check if we have resources available
            current_usage = self.get_resource_usage()
            current_cpu = current_usage.get("cpu_percent", 0)
            
            # Convert memory_mb to bytes for comparison
            memory_bytes = memory_mb * 1024 * 1024
            current_memory = current_usage.get("memory_used", 0)
            total_memory = current_usage.get("memory_total", 1)
            
            # Simple allocation strategy - just check if thresholds would be exceeded
            if ((current_cpu + cpu_percent) > self.cpu_threshold or 
                ((current_memory + memory_bytes) / total_memory * 100) > self.memory_threshold):
                logger.warning(
                    "Resource allocation denied for %s: "
                    "Requested CPU: %s%%, Memory: %sMB",
                    module_id, cpu_percent, memory_mb
                )
                return False
                
            # Store allocation
            self._module_allocations[module_id] = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "allocated_at": time.time()
            }
            logger.info(
                "Resources allocated for %s: "
                "CPU: %s%%, Memory: %sMB",
                module_id, cpu_percent, memory_mb
            )
            return True
            
    def release_resources(self, module_id: str) -> None:
        """
        Release resources allocated to a module.
        
        Args:
            module_id: ID of the module
        """
        with self._resource_lock:
            if module_id in self._module_allocations:
                allocation = self._module_allocations[module_id]
                logger.info(
                    "Resources released for %s: "
                    "CPU: %s%%, Memory: %sMB",
                    module_id, allocation['cpu_percent'], allocation['memory_mb']
                )
                del self._module_allocations[module_id]
            else:
                logger.warning("No resources were allocated for module: %s", module_id)
                
    def register_subscriber(self, callback):
        """Register a subscriber callback for resource events"""
        self._subscribers.append(callback)
        return True
        
    def notify_subscribers(self, event_type: str, data: dict):
        """Notify subscribers of resource events"""
        for callback in self._subscribers:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error("Error in subscriber callback: %s", e)
                
    def manage_memory(self) -> bool:
        """Manages memory to prevent out-of-memory errors
        
        Returns:
            bool: True if recovery action was taken, False otherwise
        """
        try:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
            
            # If memory is critically low, take action
            if available_memory < self.critical_memory_threshold:
                logger.warning("Memory critically low: %.2fMB available", available_memory)
                
                # Cross-platform memory recovery
                if platform.system() == "Windows":
                    # Windows memory recovery
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                elif platform.system() == "Linux":
                    # Linux memory recovery (less aggressive than drop_caches)
                    os.system("sync")
                    
                # Force garbage collection
                gc.collect()
                
                # Notify subscribers
                self.notify_subscribers("memory_critical", {"available_mb": available_memory})
                
                return True  # Indicate recovery action was taken
                
            return False  # No action needed
            
        except Exception as e:
            logger.error("Error managing memory: %s", e)
            return False