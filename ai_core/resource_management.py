import logging
import asyncio
import time
import os
import json
import torch
import gc
from typing import Dict, Any, Set, Callable, Optional, List
import psutil

logger = logging.getLogger(__name__)

class ResourceManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "memory_threshold": 85.0,  # percentage
            "monitoring_interval": 60,  # seconds
            "enable_memory_optimizations": True,
            "backup_interval": 1800,  # seconds (30 min)
            "checkpoint_dir": "./checkpoints",
            "resource_log_path": "./logs/resources.log"
        }
        
        self.registered_modules: Dict[str, Dict[str, Any]] = {}
        self.resource_stats: Dict[str, Any] = {
            "last_check": time.time(),
            "memory_usage": [],
            "gpu_usage": [],
            "high_memory_events": 0,
            "oom_events": 0
        }
        
        self.callbacks: Dict[str, Set[Callable]] = {
            "warning": set(),
            "critical": set(),
            "status": set(),
        }
        
        self.lock = asyncio.Lock()
        self.monitoring_task = None
        self.is_initialized = False
        self.metrics_collector = None
        
        # Ensure checkpoint directory exists
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        
        # Ensure resource log directory exists
        os.makedirs(os.path.dirname(self.config["resource_log_path"]), exist_ok=True)
        
    async def initialize(self, metrics_collector=None) -> bool:
        """Initialize the resource manager"""
        if self.is_initialized:
            return True
            
        try:
            logger.info("Initializing resource manager")
            
            # Store metrics collector reference if provided
            self.metrics_collector = metrics_collector
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitor_resources())
            
            self.is_initialized = True
            logger.info("Resource manager initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize resource manager: {e}")
            return False
    
    async def register_module(self, module_id: str, module_type: str, 
                            resource_requirements: Dict[str, Any]) -> bool:
        """Register a module with the resource manager"""
        async with self.lock:
            try:
                if module_id in self.registered_modules:
                    logger.warning(f"Module {module_id} already registered, updating configuration")
                
                # Store module information and resource requirements
                self.registered_modules[module_id] = {
                    "type": module_type,
                    "requirements": resource_requirements,
                    "last_active": time.time(),
                    "status": "active",
                    "importance": resource_requirements.get("importance", "medium"),
                    "resource_usage": {
                        "memory": 0,
                        "cpu": 0,
                        "gpu": 0
                    }
                }
                
                logger.info(f"Registered module: {module_id} (type: {module_type})")
                return True
            except Exception as e:
                logger.error(f"Failed to register module {module_id}: {e}")
                return False
    
    async def _monitor_resources(self) -> None:
        """Monitor system resources periodically"""
        try:
            while True:
                await asyncio.sleep(self.config["monitoring_interval"])
                await self._check_resources()
                await self._log_resource_usage()
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
    
    async def _check_resources(self) -> None:
        """Check system resources and take action if needed"""
        try:
            # Get memory usage
            memory_stats = psutil.virtual_memory()
            memory_percent = memory_stats.percent
            
            # Record stats
            self.resource_stats["last_check"] = time.time()
            self.resource_stats["memory_usage"].append((time.time(), memory_percent))
            
            # Keep only the most recent 100 records
            if len(self.resource_stats["memory_usage"]) > 100:
                self.resource_stats["memory_usage"].pop(0)
            
            # Check if memory usage is above threshold
            if memory_percent > self.config["memory_threshold"]:
                self.resource_stats["high_memory_events"] += 1
                
                logger.warning(f"High memory usage: {memory_percent:.1f}% > {self.config['memory_threshold']}%")
                
                # Notify warning callbacks
                await self._notify_callbacks("warning", {
                    "type": "high_memory",
                    "value": memory_percent,
                    "threshold": self.config["memory_threshold"]
                })
                
                if self.config["enable_memory_optimizations"]:
                    await self._optimize_memory_usage()
            
            # Check GPU if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        # Calculate percentage of allocated memory relative to total memory
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        gpu_percent = (allocated / total) * 100
                        
                        self.resource_stats["gpu_usage"].append((time.time(), i, gpu_percent))
                        
                        # Keep only the most recent 100 records per GPU
                        if len(self.resource_stats["gpu_usage"]) > 100 * torch.cuda.device_count():
                            self.resource_stats["gpu_usage"].pop(0)
                        
                        # Check if GPU usage is high
                        if gpu_percent > self.config["memory_threshold"]:
                            logger.warning(f"High GPU {i} memory usage: {gpu_percent:.1f}% > {self.config['memory_threshold']}%")
                            
                            # Notify warning callbacks
                            await self._notify_callbacks("warning", {
                                "type": "high_gpu_memory",
                                "device": i,
                                "value": gpu_percent,
                                "threshold": self.config["memory_threshold"]
                            })
                            
                            # Empty cache if enabled
                            if self.config["enable_memory_optimizations"]:
                                torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Error checking GPU {i} memory: {e}")
            
            # Add metrics to metrics collector if available
            if self.metrics_collector is not None:
                timestamp = time.time()
                try:
                    # Add memory metrics
                    await self.metrics_collector._add_metric_point("memory_percent", memory_percent, timestamp)
                    await self.metrics_collector._add_metric_point("memory_available", memory_stats.available, timestamp, {"total": memory_stats.total})
                    
                    # Add CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    await self.metrics_collector._add_metric_point("cpu_percent", cpu_percent, timestamp)
                    
                    # Add disk metrics
                    disk = psutil.disk_usage('/')
                    await self.metrics_collector._add_metric_point("disk_percent", disk.percent, timestamp)
                    await self.metrics_collector._add_metric_point("disk_free", disk.free, timestamp, {"total": disk.total})
                    
                    # Add module-specific metrics
                    for module_id, module_info in self.registered_modules.items():
                        if module_info["status"] == "active":
                            await self.metrics_collector._add_metric_point(
                                f"module_{module_id}_memory", 
                                module_info["resource_usage"].get("memory", 0), 
                                timestamp
                            )
                except Exception as e:
                    logger.error(f"Failed to add metrics to collector: {e}")
        
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
    
    async def _optimize_memory_usage(self) -> None:
        """Optimize memory usage by suspending low-importance modules"""
        async with self.lock:
            # Sort modules by importance (low to high)
            modules_by_importance = sorted(
                [(k, v) for k, v in self.registered_modules.items()],
                key=lambda x: {
                    "low": 0,
                    "medium": 1,
                    "high": 2,
                    "critical": 3
                }.get(x[1]["importance"], 1)
            )
            
            # Suspend low importance modules first
            for module_id, module_info in modules_by_importance:
                if module_info["status"] == "active" and module_info["importance"] in ["low", "medium"]:
                    await self._suspend_module(module_id)
                    
                    # Check if memory is now below threshold
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent < self.config["memory_threshold"] - 10:  # Add some buffer
                        break
    
    async def _suspend_module(self, module_id: str) -> None:
        """Suspend a module to free resources"""
        if module_id not in self.registered_modules:
            return
            
        try:
            # Update module status
            self.registered_modules[module_id]["status"] = "suspended"
            
            # Notify module about suspension
            await self._notify_callbacks("status", {
                "type": "module_suspended",
                "module_id": module_id
            })
            
            logger.info(f"Suspended module {module_id} to free resources")
            
        except Exception as e:
            logger.error(f"Error suspending module {module_id}: {e}")
    
    async def _resume_module(self, module_id: str) -> None:
        """Resume a previously suspended module"""
        if module_id not in self.registered_modules or self.registered_modules[module_id]["status"] != "suspended":
            return
            
        try:
            # Update module status
            self.registered_modules[module_id]["status"] = "active"
            self.registered_modules[module_id]["last_active"] = time.time()
            
            # Notify module about resumption
            await self._notify_callbacks("status", {
                "type": "module_resumed",
                "module_id": module_id
            })
            
            logger.info(f"Resumed module {module_id}")
            
        except Exception as e:
            logger.error(f"Error resuming module {module_id}: {e}")
    
    async def get_module_status(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a registered module"""
        if module_id not in self.registered_modules:
            return None
        
        return self.registered_modules[module_id]
    
    async def update_module_usage(self, module_id: str, resource_usage: Dict[str, float]) -> bool:
        """Update resource usage for a module"""
        if module_id not in self.registered_modules:
            return False
            
        try:
            async with self.lock:
                self.registered_modules[module_id]["resource_usage"] = resource_usage
                self.registered_modules[module_id]["last_active"] = time.time()
            return True
        except Exception as e:
            logger.error(f"Error updating module usage: {e}")
            return False
    
    async def _log_resource_usage(self) -> None:
        """Log resource usage to file"""
        try:
            log_entry = {
                "timestamp": time.time(),
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "modules": {
                    module_id: {
                        "status": info["status"],
                        "resource_usage": info["resource_usage"]
                    }
                    for module_id, info in self.registered_modules.items()
                }
            }
            
            # Add GPU info if available
            if torch.cuda.is_available():
                log_entry["gpu"] = {}
                for i in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        log_entry["gpu"][f"gpu_{i}"] = {
                            "percent": (allocated / total) * 100,
                            "allocated_mb": allocated / (1024 * 1024),
                            "total_mb": total / (1024 * 1024)
                        }
                    except Exception:
                        pass
            
            # Append to log file
            with open(self.config["resource_log_path"], "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging resource usage: {e}")
    
    async def get_module_resource_metrics(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed resource metrics for a specific module"""
        if module_id not in self.registered_modules:
            return None
            
        try:
            module_info = self.registered_modules[module_id]
            
            metrics = {
                "status": module_info["status"],
                "importance": module_info["importance"],
                "resource_usage": module_info["resource_usage"],
                "last_active": time.time() - module_info["last_active"],
                "system": {
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_percent": psutil.cpu_percent()
                }
            }
            
            # Add GPU metrics if relevant
            if torch.cuda.is_available() and "gpu" in module_info["resource_usage"]:
                device_id = module_info["resource_usage"].get("gpu_device", 0)
                if 0 <= device_id < torch.cuda.device_count():
                    allocated = torch.cuda.memory_allocated(device_id)
                    total = torch.cuda.get_device_properties(device_id).total_memory
                    metrics["gpu"] = {
                        "device_id": device_id,
                        "percent": (allocated / total) * 100,
                        "allocated_mb": allocated / (1024 * 1024),
                        "total_mb": total / (1024 * 1024)
                    }
                    
            return metrics
        except Exception as e:
            logger.error(f"Error getting resource metrics for module {module_id}: {e}")
            return None
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for resource events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = set()
            
        self.callbacks[event_type].add(callback)
        
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify all callbacks for a specific event type"""
        if event_type not in self.callbacks:
            return
            
        for callback in self.callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")
    
    async def create_backup(self, module_id: str, data: Any) -> str:
        """Create a backup of module state"""
        if module_id not in self.registered_modules:
            raise ValueError(f"Module {module_id} not registered")
        
        try:
            timestamp = int(time.time())
            backup_dir = os.path.join(self.config["checkpoint_dir"], module_id)
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_path = os.path.join(backup_dir, f"backup_{timestamp}.pkl")
            
            # Save backup data
            torch.save(data, backup_path)
            
            # Update metadata
            metadata_path = os.path.join(backup_dir, "metadata.json")
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {"backups": []}
                
                metadata["backups"].append({
                    "path": backup_path,
                    "timestamp": timestamp,
                    "size": os.path.getsize(backup_path)
                })
                
                # Limit number of backups
                if len(metadata["backups"]) > 5:
                    oldest = metadata["backups"][0]
                    if os.path.exists(oldest["path"]):
                        os.remove(oldest["path"])
                    metadata["backups"] = metadata["backups"][1:]
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error updating backup metadata: {e}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup for {module_id}: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shut down resource manager"""
        logger.info("Shutting down resource manager")
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Final log of resource usage
        await self._log_resource_usage()
        
        # Run garbage collection
        gc.collect()
        
        # Empty CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Resource manager shutdown complete")
