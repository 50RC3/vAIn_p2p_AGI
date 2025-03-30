import logging
import torch
import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Callable
import os
import json

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Centralized resource management for AI modules.
    
    This class handles:
    - Memory management (RAM and VRAM)
    - Model sharing and coordination
    - Resource allocation strategies
    - Backup and restore functionality
    """
    
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
        
        self.monitor_task = None
        self.is_initialized = False
        self.lock = asyncio.Lock()
        
        # Create checkpoint directory
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up dedicated resource logging"""
        os.makedirs(os.path.dirname(self.config["resource_log_path"]), exist_ok=True)
        
        # Create file handler for resource logs
        file_handler = logging.FileHandler(self.config["resource_log_path"])
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Get logger and add handler
        resource_logger = logging.getLogger("ai_core.resources")
        resource_logger.setLevel(logging.INFO)
        for handler in resource_logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_handler.baseFilename:
                break
        else:
            resource_logger.addHandler(file_handler)
    
    async def initialize(self):
        """Initialize the resource manager and start monitoring"""
        try:
            logger.info("Initializing resource manager")
            
            # Get initial resource stats
            await self._update_resource_stats()
            
            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_resources())
            
            self.is_initialized = True
            logger.info("Resource manager initialized successfully")
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
                    "importance": resource_requirements.get("importance", "medium")
                }
                
                logger.info(f"Registered module: {module_id} (type: {module_type})")
                return True
            except Exception as e:
                logger.error(f"Failed to register module {module_id}: {e}")
                return False
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for resource events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify registered callbacks about events"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in resource callback: {e}")
    
    async def _monitor_resources(self) -> None:
        """Monitor system resources periodically"""
        try:
            while True:
                await asyncio.sleep(self.config["monitoring_interval"])
                
                # Check resources and take action if needed
                high_usage = await self._check_resources()
                
                if high_usage:
                    # Execute resource optimization strategy
                    await self._optimize_resources()
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
    
    async def _check_resources(self) -> bool:
        """Check system resources and return True if usage is high"""
        try:
            # Update resource statistics
            await self._update_resource_stats()
            
            # Get current memory usage
            memory_usage = self.resource_stats.get("current_memory_percent", 0)
            gpu_memory = self.resource_stats.get("current_gpu_percent", 0)
            
            # Check if memory usage is high
            threshold = self.config["memory_threshold"]
            is_high_usage = memory_usage > threshold or gpu_memory > threshold
            
            if is_high_usage:
                self.resource_stats["high_memory_events"] += 1
                await self._notify_callbacks("warning", {
                    "type": "high_resource_usage",
                    "memory_percent": memory_usage,
                    "gpu_percent": gpu_memory,
                    "threshold": threshold
                })
                
                # Log the high usage event
                logger.warning(f"High resource usage: RAM {memory_usage:.1f}%, GPU {gpu_memory:.1f}%")
            
            return is_high_usage
        
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return False
    
    async def _update_resource_stats(self) -> None:
        """Update resource statistics"""
        try:
            # Get system memory usage
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get GPU memory if available
            gpu_percent = 0
            if torch.cuda.is_available():
                try:
                    # Calculate average GPU usage across all devices
                    total_usage = 0
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        total_usage += (allocated / total) * 100
                    
                    gpu_percent = total_usage / torch.cuda.device_count()
                except Exception as e:
                    logger.error(f"Error getting GPU memory: {e}")
            
            # Update current stats
            self.resource_stats["current_memory_percent"] = memory_percent
            self.resource_stats["current_gpu_percent"] = gpu_percent
            self.resource_stats["last_check"] = time.time()
            
            # Add to history (keep only last 10 readings)
            self.resource_stats["memory_usage"].append(memory_percent)
            self.resource_stats["gpu_usage"].append(gpu_percent)
            
            # Trim history
            if len(self.resource_stats["memory_usage"]) > 10:
                self.resource_stats["memory_usage"] = self.resource_stats["memory_usage"][-10:]
                self.resource_stats["gpu_usage"] = self.resource_stats["gpu_usage"][-10:]
            
        except Exception as e:
            logger.error(f"Error updating resource stats: {e}")
    
    async def _optimize_resources(self) -> None:
        """Optimize resource usage based on current state"""
        try:
            logger.info("Optimizing resources due to high usage")
            
            # Clear PyTorch cache
            if torch.cuda.is_available() and self.config["enable_memory_optimizations"]:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
            # Sort modules by importance
            modules_by_importance = sorted(
                self.registered_modules.items(),
                key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x[1]["importance"], 0)
            )
            
            # Release resources for low-importance modules
            for module_id, info in modules_by_importance:
                if info["importance"] == "low" and info["status"] == "active":
                    await self._suspend_module(module_id)
            
            # Notify about optimization
            await self._notify_callbacks("status", {
                "type": "resources_optimized",
                "memory_before": self.resource_stats["memory_usage"][-2] if len(self.resource_stats["memory_usage"]) > 1 else None,
                "memory_after": self.resource_stats["memory_usage"][-1],
            })
            
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
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
    
    async def request_resources(self, module_id: str, requirements: Dict[str, Any]) -> bool:
        """Request resources for a specific operation"""
        if module_id not in self.registered_modules:
            logger.warning(f"Unregistered module {module_id} requested resources")
            return False
        
        try:
            # Update last active timestamp
            self.registered_modules[module_id]["last_active"] = time.time()
            
            # Check if we need to optimize before allocation
            memory_usage = self.resource_stats.get("current_memory_percent", 0)
            threshold = self.config["memory_threshold"]
            
            # If memory usage is already high, try to optimize first
            if memory_usage > threshold * 0.9:  # 90% of threshold
                await self._optimize_resources()
            
            # If module was suspended, resume it
            if self.registered_modules[module_id]["status"] == "suspended":
                await self._resume_module(module_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling resource request from {module_id}: {e}")
            return False
    
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
                    "module_id": module_id
                })
                
                # Keep only the latest 5 backups in metadata
                if len(metadata["backups"]) > 5:
                    metadata["backups"] = sorted(
                        metadata["backups"], 
                        key=lambda x: x["timestamp"], 
                        reverse=True
                    )[:5]
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error updating backup metadata: {e}")
            
            logger.info(f"Created backup for {module_id}: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup for {module_id}: {e}")
            raise
    
    async def restore_backup(self, module_id: str, backup_path: Optional[str] = None) -> Optional[Any]:
        """Restore module state from backup"""
        if module_id not in self.registered_modules:
            raise ValueError(f"Module {module_id} not registered")
        
        try:
            if backup_path is None:
                # Find the latest backup
                metadata_path = os.path.join(self.config["checkpoint_dir"], module_id, "metadata.json")
                
                if not os.path.exists(metadata_path):
                    logger.warning(f"No backup metadata found for {module_id}")
                    return None
                
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                if not metadata.get("backups"):
                    logger.warning(f"No backups found for {module_id}")
                    return None
                
                # Get latest backup
                backups = sorted(metadata["backups"], key=lambda x: x["timestamp"], reverse=True)
                backup_path = backups[0]["path"]
            
            # Load backup data
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return None
                
            data = torch.load(backup_path)
            logger.info(f"Restored backup for {module_id}: {backup_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error restoring backup for {module_id}: {e}")
            return None
    
    async def shutdown(self):
        """Gracefully shut down resource manager"""
        try:
            logger.info("Shutting down resource manager")
            
            # Cancel monitoring task
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Final resource stats update
            await self._update_resource_stats()
            
            # Save final resource stats
            stats_path = os.path.join(self.config["checkpoint_dir"], "resource_stats.json")
            with open(stats_path, "w") as f:
                json.dump({
                    k: v for k, v in self.resource_stats.items() 
                    if k not in ["memory_usage", "gpu_usage"] or len(v) <= 10
                }, f, indent=2)
                
            logger.info("Resource manager shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down resource manager: {e}")
