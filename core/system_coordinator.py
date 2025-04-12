"""
System Coordinator - Coordinates interactions between system components
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set, Callable
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemCoordinator:
    """Coordinates cross-module interactions and resource sharing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize system coordinator.
        
        Args:
            config: Configuration dictionary for the coordinator
        """
        self.config = config or {
            "status_check_interval": 60,  # seconds
            "resource_monitor_enabled": True,
            "cross_module_events_enabled": True,
            "log_directory": "logs",
            "status_directory": "status"
        }
        
        # Components registered with coordinator
        self.components: Dict[str, Any] = {}
        
        # Event listeners for cross-module communication
        self.event_listeners: Dict[str, Set[Callable]] = {}
        
        # Status tracking
        self.system_status = {
            "initialized": False,
            "last_status_check": 0,
            "components": {},
            "resources": {}
        }
        
        # Tasks
        self.status_task = None
        self.running = False
        self._lock = asyncio.Lock()
        
        # Create necessary directories
        os.makedirs(self.config["log_directory"], exist_ok=True)
        os.makedirs(self.config["status_directory"], exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize the coordinator and register core components."""
        try:
            logger.info("Initializing System Coordinator")
            self.running = True
            
            # Start status checking task
            self.status_task = asyncio.create_task(self._check_status_periodically())
            
            # Initialize resource monitoring if enabled
            if self.config["resource_monitor_enabled"]:
                await self._setup_resource_monitoring()
            
            self.system_status["initialized"] = True
            logger.info("System Coordinator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize System Coordinator: {e}")
            return False
    
    async def _setup_resource_monitoring(self) -> None:
        """Set up resource monitoring."""
        try:
            # Try to import and set up memory monitoring
            from network.memory_monitor import MemoryMonitor
            
            # Create monitor with callback
            memory_monitor = MemoryMonitor(threshold=0.8, check_interval=30)
            await memory_monitor.start_monitoring(
                callback=lambda usage: self.dispatch_event("high_memory", {"usage": usage})
            )
            
            # Register with components
            self.register_component("memory_monitor", memory_monitor)
            logger.info("Memory monitoring initialized")
            
        except ImportError:
            logger.warning("Memory monitor module not available")
            
        try:
            # Try to access the resource manager if available
            from ai_core.resource_management import ResourceManager
            
            # Register callback with resource manager
            if "resource_manager" in self.components:
                resource_manager = self.components["resource_manager"]
                resource_manager.register_callback(
                    "warning", 
                    lambda data: self.dispatch_event("resource_warning", data)
                )
                logger.info("Resource manager callbacks registered")
        except ImportError:
            logger.warning("Resource manager module not available")
    
    def register_component(self, component_id: str, component: Any) -> bool:
        """Register a component with the coordinator."""
        if component_id in self.components:
            logger.warning(f"Component {component_id} already registered, will be replaced")
            
        self.components[component_id] = component
        self.system_status["components"][component_id] = {
            "registered_at": time.time(),
            "active": True
        }
        
        logger.info(f"Registered component: {component_id}")
        return True
    
    def register_event_listener(self, event_type: str, listener: Callable) -> None:
        """Register a listener for a specific event type."""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = set()
            
        self.event_listeners[event_type].add(listener)
        logger.debug(f"Registered listener for event type: {event_type}")
    
    def dispatch_event(self, event_type: str, data: Any) -> None:
        """Dispatch an event to all registered listeners."""
        if not self.config["cross_module_events_enabled"]:
            return
            
        if event_type not in self.event_listeners:
            return
            
        for listener in self.event_listeners[event_type]:
            try:
                # Handle both synchronous and asynchronous listeners
                if asyncio.iscoroutinefunction(listener):
                    # Create task for async listeners
                    asyncio.create_task(listener(data))
                else:
                    listener(data)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        async with self._lock:
            # Update resource statistics
            await self._update_resource_stats()
            
            # Update component status
            for component_id, component in self.components.items():
                if hasattr(component, "get_status") and callable(component.get_status):
                    try:
                        if asyncio.iscoroutinefunction(component.get_status):
                            self.system_status["components"][component_id]["status"] = await component.get_status()
                        else:
                            self.system_status["components"][component_id]["status"] = component.get_status()
                    except Exception as e:
                        logger.error(f"Error getting status for {component_id}: {e}")
                
            return self.system_status
    
    async def _check_status_periodically(self) -> None:
        """Periodically check and update system status."""
        while self.running:
            try:
                # Get current status
                await self.get_status()
                
                # Save status to file
                self._save_status()
                
                # Check for any critical issues
                await self._check_for_issues()
                
                # Sleep for interval
                await asyncio.sleep(self.config["status_check_interval"])
                
            except asyncio.CancelledError:
                logger.info("Status checking task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in status checking: {e}")
                await asyncio.sleep(10)  # Sleep a bit before retrying
    
    def _save_status(self) -> None:
        """Save current status to file."""
        try:
            import json
            status_path = Path(self.config["status_directory"]) / "system_status.json"
            
            # Write compact status (exclude large objects)
            compact_status = {
                "initialized": self.system_status["initialized"],
                "last_check": time.time(),
                "components": {
                    k: {"active": v.get("active", False)}
                    for k, v in self.system_status["components"].items()
                },
                "resources": self.system_status["resources"]
            }
            
            with open(status_path, "w") as f:
                json.dump(compact_status, f)
                
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    async def _update_resource_stats(self) -> None:
        """Update resource statistics."""
        try:
            import psutil
            
            self.system_status["resources"] = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
            
            # Add GPU stats if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_stats = []
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        gpu_stats.append({
                            "id": i,
                            "usage_percent": (allocated / total) * 100,
                            "allocated_mb": allocated / (1024 * 1024)
                        })
                    self.system_status["resources"]["gpu"] = gpu_stats
            except (ImportError, Exception):
                pass
                
        except ImportError:
            logger.warning("psutil not available, resource stats will be limited")
        except Exception as e:
            logger.error(f"Error updating resource stats: {e}")
    
    async def _check_for_issues(self) -> None:
        """Check for critical issues in the system."""
        resources = self.system_status["resources"]
        
        # Check for high resource usage
        if resources.get("memory_usage", 0) > 90:
            self.dispatch_event("critical_memory", {
                "usage": resources["memory_usage"],
                "timestamp": time.time()
            })
            
        if resources.get("cpu_usage", 0) > 95:
            self.dispatch_event("critical_cpu", {
                "usage": resources["cpu_usage"],
                "timestamp": time.time()
            })
    
    async def coordinate_resources(self) -> bool:
        """Coordinate resource allocation across components."""
        if "resource_manager" not in self.components:
            logger.warning("No resource manager available for coordination")
            return False
            
        try:
            resource_manager = self.components["resource_manager"]
            
            # Get resources from all monitoring components
            resource_stats = {}
            
            # Memory monitor
            if "memory_monitor" in self.components:
                memory_monitor = self.components["memory_monitor"]
                if hasattr(memory_monitor, "get_memory_usage"):
                    resource_stats["memory"] = memory_monitor.get_memory_usage()
                    
            # Update resource manager with stats
            if hasattr(resource_manager, "update_resources"):
                await resource_manager.update_resources(resource_stats)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Resource coordination failed: {e}")
            return False
            
    async def shutdown(self) -> bool:
        """Shutdown the coordinator and all registered components."""
        logger.info("Shutting down System Coordinator")
        self.running = False
        
        if self.status_task and not self.status_task.done():
            self.status_task.cancel()
            try:
                await self.status_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components in reverse registration order
        component_ids = list(self.components.keys())
        for component_id in reversed(component_ids):
            component = self.components[component_id]
            try:
                if hasattr(component, "shutdown") and callable(component.shutdown):
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown()
                    logger.info(f"Component {component_id} shutdown successfully")
            except Exception as e:
                logger.error(f"Error shutting down component {component_id}: {e}")
                
        logger.info("System Coordinator shutdown complete")
        return True


# Helper function to get or create the system coordinator
_coordinator_instance = None

def get_coordinator(config: Optional[Dict[str, Any]] = None) -> SystemCoordinator:
    """Get or create the system coordinator singleton instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = SystemCoordinator(config)
    return _coordinator_instance


async def initialize_system(config: Optional[Dict[str, Any]] = None) -> SystemCoordinator:
    """Initialize the system with the coordinator."""
    coordinator = get_coordinator(config)
    await coordinator.initialize()
    return coordinator
