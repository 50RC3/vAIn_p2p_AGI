import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Type, Callable
import json
import os
import inspect
import threading

from .resource_management import ResourceManager

logger = logging.getLogger(__name__)

class ModuleRegistryError(Exception):
    """Base exception for module registry errors"""
    pass

class ModuleRegistry:
    """
    Central registry for AI modules with dependency management.
    
    This class provides:
    - Module registration and discovery
    - Dependency resolution
    - Cross-module coordination
    - Lifecycle management
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ModuleRegistry()
        return cls._instance
    
    def __init__(self):
        if self.__class__._instance is not None:
            raise RuntimeError("ModuleRegistry is a singleton. Use ModuleRegistry.get_instance()")
        
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.module_types: Dict[str, str] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.startup_order: List[str] = []
        self.resource_manager: Optional[ResourceManager] = None
        self.metrics_collector = None
        self.is_initialized = False
        self.config_path = os.path.join("config", "modules.json")
        self.registry_lock = asyncio.Lock()
        self.callbacks: Dict[str, Set[Callable]] = {
            "module_added": set(),
            "module_removed": set(),
            "initialization_complete": set(),
            "error": set()
        }
    
    async def initialize(self, resource_manager: Optional[ResourceManager] = None,
                       metrics_collector=None) -> bool:
        """Initialize the module registry"""
        async with self.registry_lock:
            if self.is_initialized:
                logger.warning("Module registry already initialized")
                return True
                
            try:
                logger.info("Initializing module registry")
                
                # Store metrics collector reference
                self.metrics_collector = metrics_collector
                
                # Set resource manager
                self.resource_manager = resource_manager or ResourceManager()
                if not resource_manager:
                    await self.resource_manager.initialize(metrics_collector=metrics_collector)
                
                # Load configuration if exists
                await self._load_configuration()
                
                # Compute module startup order based on dependencies
                self._compute_startup_order()
                
                self.is_initialized = True
                logger.info("Module registry initialized")
                
                # Notify initialization complete
                await self._notify_callbacks("initialization_complete", {})
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize module registry: {e}")
                await self._notify_callbacks("error", {
                    "phase": "initialization",
                    "error": str(e)
                })
                return False
    
    async def _load_configuration(self):
        """Load module configuration from file"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                logger.info(f"Loading module configuration from {self.config_path}")
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                
                # Register modules from configuration
                for module_id, module_config in config.get("modules", {}).items():
                    self.modules[module_id] = module_config
                    self.dependencies[module_id] = module_config.get("dependencies", [])
            
        except Exception as e:
            logger.error(f"Error loading module configuration: {e}")
    
    async def _save_configuration(self):
        """Save current module configuration to file"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config = {
                "modules": self.modules,
                "last_updated": time.time()
            }
            
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved module configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving module configuration: {e}")
    
    def _compute_startup_order(self):
        """Compute optimal startup order based on dependencies"""
        visited: Set[str] = set()
        startup_order: List[str] = []
        
        def visit(module_id: str, path: Set[str] = None):
            if path is None:
                path = set()
                
            if module_id in path:
                raise ModuleRegistryError(f"Circular dependency detected: {' -> '.join(path)} -> {module_id}")
                
            if module_id in visited:
                return
                
            path.add(module_id)
            
            # Visit dependencies first
            for dependency in self.dependencies.get(module_id, []):
                if dependency not in self.modules:
                    logger.warning(f"Missing dependency: {dependency} for module {module_id}")
                    continue
                visit(dependency, path.copy())
                
            visited.add(module_id)
            startup_order.append(module_id)
            
        # Visit all modules
        for module_id in self.modules:
            if module_id not in visited:
                visit(module_id)
                
        self.startup_order = startup_order
        logger.info(f"Computed startup order: {', '.join(startup_order)}")
    
    async def register_module(self, module_id: str, module_class: Type, 
                            dependencies: List[str] = None, 
                            config: Dict[str, Any] = None,
                            replace: bool = False) -> bool:
        """Register a module with the registry"""
        async with self.registry_lock:
            try:
                if module_id in self.modules:
                    if replace:
                        logger.info(f"Replacing existing module: {module_id}")
                    else:
                        logger.warning(f"Module {module_id} already registered")
                        return True
                    
                # Register module
                self.modules[module_id] = {
                    "class": module_class.__name__,
                    "module": module_class.__module__,
                    "config": config or {},
                    "status": "registered",
                    "registered_time": time.time()
                }
                
                # Register dependencies
                self.dependencies[module_id] = dependencies or []
                
                # Update startup order
                self._compute_startup_order()
                
                # Save configuration
                await self._save_configuration()
                
                # Register with resource manager if available
                if self.resource_manager:
                    resource_requirements = config.get("resource_requirements", {})
                    await self.resource_manager.register_module(
                        module_id, 
                        module_class.__name__, 
                        resource_requirements
                    )
                
                # Track module registration in metrics if available
                if self.metrics_collector:
                    try:
                        timestamp = time.time()
                        await self.metrics_collector._add_metric_point(
                            "module_registrations",
                            1.0,  # Count
                            timestamp,
                            {"module_id": module_id, "class": module_class.__name__}
                        )
                        await self.metrics_collector._add_metric_point(
                            f"module_{module_id}_status",
                            1.0,  # 1 = registered
                            timestamp,
                            {"status": "registered"}
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add module registration metric: {e}")
                
                # Notify callbacks
                await self._notify_callbacks("module_added", {
                    "module_id": module_id,
                    "class": module_class.__name__
                })
                
                logger.info(f"Registered module: {module_id} ({module_class.__name__})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register module {module_id}: {e}")
                await self._notify_callbacks("error", {
                    "operation": "register",
                    "module_id": module_id,
                    "error": str(e)
                })
                return False
    
    async def unregister_module(self, module_id: str) -> bool:
        """Unregister a module from the registry"""
        async with self.registry_lock:
            if module_id not in self.modules:
                logger.warning(f"Module {module_id} not found in registry")
                return False
                
            try:
                # Check if any modules depend on this one
                dependent_modules = []
                for other_id, deps in self.dependencies.items():
                    if module_id in deps:
                        dependent_modules.append(other_id)
                
                if dependent_modules:
                    raise ModuleRegistryError(
                        f"Cannot unregister module {module_id}: still required by {', '.join(dependent_modules)}"
                    )
                
                # Remove module
                module_info = self.modules.pop(module_id)
                self.dependencies.pop(module_id, None)
                
                # Update startup order
                self._compute_startup_order()
                
                # Save configuration
                await self._save_configuration()
                
                # Notify callbacks
                await self._notify_callbacks("module_removed", {
                    "module_id": module_id,
                    "class": module_info.get("class")
                })
                
                logger.info(f"Unregistered module: {module_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister module {module_id}: {e}")
                await self._notify_callbacks("error", {
                    "operation": "unregister",
                    "module_id": module_id,
                    "error": str(e)
                })
                return False
    
    async def update_module_status(self, module_id: str, status: str) -> bool:
        """Update the status of a registered module"""
        if module_id not in self.modules:
            logger.warning(f"Cannot update status for unregistered module {module_id}")
            return False
            
        try:
            self.modules[module_id]["status"] = status
            self.modules[module_id]["last_status_update"] = time.time()
            
            # Track status change in metrics if available
            if self.metrics_collector:
                try:
                    timestamp = time.time()
                    status_code = {
                        "registered": 1.0,
                        "initializing": 2.0,
                        "active": 3.0,
                        "suspended": 4.0,
                        "error": 5.0,
                        "terminated": 6.0
                    }.get(status, 0.0)
                    
                    await self.metrics_collector._add_metric_point(
                        f"module_{module_id}_status",
                        status_code,
                        timestamp,
                        {"status": status}
                    )
                except Exception as e:
                    logger.warning(f"Failed to add module status metric: {e}")
            
            # Save configuration
            await self._save_configuration()
            
            # Notify callbacks of status change
            await self._notify_callbacks("module_status_change", {
                "module_id": module_id,
                "status": status,
                "timestamp": time.time()
            })
            
            logger.info(f"Updated status for module {module_id} to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update module status: {e}")
            return False
    
    async def get_module_info(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered module"""
        if module_id not in self.modules:
            return None
            
        module_info = self.modules[module_id].copy()
        
        # Add dependency information
        module_info["dependencies"] = self.dependencies.get(module_id, [])
        
        # Add resource information if available
        if self.resource_manager:
            resource_info = await self.resource_manager.get_module_status(module_id)
            if resource_info:
                module_info["resources"] = resource_info
                
        return module_info
    
    async def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered modules"""
        result = {}
        for module_id in self.modules:
            result[module_id] = await self.get_module_info(module_id)
        return result
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics and module status"""
        metrics = {
            "timestamp": time.time(),
            "modules": {
                "total": len(self.modules),
                "active": sum(1 for info in self.modules.values() if info.get("status") == "active"),
                "error": sum(1 for info in self.modules.values() if info.get("status") == "error"),
                "suspended": sum(1 for info in self.modules.values() if info.get("status") == "suspended")
            },
            "startup_order_length": len(self.startup_order)
        }
        
        # Add resource metrics if available
        if self.resource_manager:
            try:
                import psutil
                metrics["system"] = {
                    "memory": {
                        "percent": psutil.virtual_memory().percent,
                        "available": psutil.virtual_memory().available,
                        "total": psutil.virtual_memory().total
                    },
                    "cpu": {
                        "percent": psutil.cpu_percent(),
                        "count": psutil.cpu_count()
                    },
                    "disk": {
                        "percent": psutil.disk_usage('/').percent,
                        "free": psutil.disk_usage('/').free,
                        "total": psutil.disk_usage('/').total
                    }
                }
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to get system metrics: {e}")
        
        return metrics
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for registry events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify registered callbacks"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in registry callback: {e}")
    
    async def check_dependencies(self, module_id: str) -> Dict[str, bool]:
        """Check if all dependencies for a module are available"""
        if module_id not in self.modules:
            raise ModuleRegistryError(f"Module {module_id} not registered")
            
        result = {}
        for dependency in self.dependencies.get(module_id, []):
            result[dependency] = dependency in self.modules
            
        return result
    
    async def initialize_modules(self):
        """Initialize all registered modules in dependency order."""
        if self.is_initialized:
            logger.warning("Modules already initialized")
            return
        
        for module_name in self.startup_order:
            module = self.modules[module_name]
            if hasattr(module, 'initialize'):
                try:
                    init_method = getattr(module, 'initialize')
                    if inspect.iscoroutinefunction(init_method):
                        await init_method()
                    else:
                        init_method()
                    logger.info(f"Initialized module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize module {module_name}: {e}")
        
        self.is_initialized = True
    
    async def shutdown_modules(self):
        """Shutdown all registered modules in reverse dependency order."""
        if not self.is_initialized:
            logger.warning("Modules not initialized, nothing to shut down")
            return
            
        # Shutdown in reverse order
        for module_name in reversed(self.startup_order):
            module = self.modules[module_name]
            if hasattr(module, 'shutdown'):
                try:
                    shutdown_method = getattr(module, 'shutdown')
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.info(f"Shutdown module: {module_name}")
                except Exception as e:
                    logger.error(f"Error shutting down module {module_name}: {e}")
                    
        self.is_initialized = False
    
    async def shutdown(self):
        """Gracefully shut down module registry"""
        try:
            logger.info("Shutting down module registry")
            
            # Update status for all modules
            for module_id in self.modules:
                await self.update_module_status(module_id, "terminated")
            
            # Save configuration
            await self._save_configuration()
            
            # Track shutdown in metrics if available
            if self.metrics_collector:
                try:
                    timestamp = time.time()
                    await self.metrics_collector._add_metric_point(
                        "registry_shutdown",
                        1.0,
                        timestamp,
                        {"modules_count": len(self.modules)}
                    )
                except Exception as e:
                    logger.warning(f"Failed to add shutdown metric: {e}")
            
            # Shut down resource manager if we created it
            if self.resource_manager and hasattr(self.resource_manager, "shutdown"):
                await self.resource_manager.shutdown()
                
            logger.info("Module registry shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down module registry: {e}")
