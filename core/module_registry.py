import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Type, Callable
import json
import os
import inspect
import threading

try:
    from .resource_management import ResourceManager
except ImportError:
    # Define a placeholder ResourceManager if the module is not available
    class ResourceManager:
        """Placeholder for ResourceManager when the actual module is not available"""
    logging.getLogger(__name__).warning("ResourceManager module not found, using placeholder")

logger = logging.getLogger(__name__)

class ModuleRegistryError(Exception):
    """Base exception for module registry errors"""

class ConfigManager:
    """Handles loading and saving module configuration"""
    def __init__(self, config_path: str):
        self.config_path = config_path

    async def load_configuration(self) -> Dict[str, Any]:
        """
        Loads configuration from a JSON file.
        Returns an empty dictionary if the file does not exist or an error occurs.
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            if os.path.exists(self.config_path):
                logger.info("Loading module configuration from %s", self.config_path)
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return config
        except (OSError, IOError) as e:
            logger.error("Error accessing configuration file: %s", e)
        except json.JSONDecodeError as e:
            logger.error("Error parsing configuration file: %s", e)
        return {}

    async def save_configuration(self, modules: Dict[str, Any]) -> None:
        """
        Saves the current module configuration to a JSON file.
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            config = {
                "modules": modules,
                "last_updated": time.time()
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info("Saved module configuration to %s", self.config_path)
        except (OSError, IOError) as e:
            logger.error("Error saving module configuration file: %s", e)
        except json.JSONEncodeError as e:
            logger.error("Error encoding module configuration as JSON: %s", e)
        except TypeError as e:
            logger.error("Error with module configuration data types: %s", e)

class DependencyResolver:
    """Resolves module dependencies and computes startup order"""
    def __init__(self, modules: Dict[str, Any], dependencies: Dict[str, List[str]]):
        self.modules = modules
        self.dependencies = dependencies
        self.startup_order: List[str] = []

    def compute_startup_order(self) -> List[str]:
        """
        Computes the startup order of modules based on their dependencies.
        Raises an error if circular dependencies are detected.
        """
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
            for dependency in self.dependencies.get(module_id, []):
                if dependency not in self.modules:
                    logger.warning("Missing dependency: %s for module %s", dependency, module_id)
                    continue
                visit(dependency, path.copy())
            visited.add(module_id)
            startup_order.append(module_id)

        for module_id in self.modules:
            if module_id not in visited:
                visit(module_id)
        self.startup_order = startup_order
        logger.info("Computed startup order: %s", ", ".join(startup_order))
        return startup_order

class LifecycleManager:
    """Manages initialization and shutdown of modules"""
    def __init__(self, modules: Dict[str, Any], startup_order: List[str]):
        self.modules = modules
        self.startup_order = startup_order
        self.is_initialized = False

    async def initialize_modules(self):
        """
        Initializes all modules in the computed startup order.
        """
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
                    logger.info("Initialized module: %s", module_name)
                except (TypeError, ValueError, AttributeError, RuntimeError, ModuleRegistryError) as e:
                    logger.error(f"Failed to initialize module {module_name}: {e}")
        self.is_initialized = True

    async def shutdown_modules(self):
        """
        Shuts down all modules in reverse startup order.
        """
        if not self.is_initialized:
            logger.warning("Modules not initialized, nothing to shut down")
            return
        for module_name in reversed(self.startup_order):
            module = self.modules[module_name]
            if hasattr(module, 'shutdown'):
                try:
                    shutdown_method = getattr(module, 'shutdown')
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.info("Shutdown module: %s", module_name)
                except (TypeError, ValueError, AttributeError, RuntimeError, ModuleRegistryError, asyncio.TimeoutError) as e:
                    logger.error(f"Error shutting down module {module_name}: {e}")
        self.is_initialized = False

class MetricsTracker:
    """Tracks metrics related to modules and system"""
    def __init__(self, modules: Dict[str, Any], resource_manager: Optional[ResourceManager], metrics_collector: Any):
        self.modules = modules
        self.resource_manager = resource_manager
        self.metrics_collector = metrics_collector

    async def track_registration(self, module_id: str, module_class: Type):
        """
        Tracks metrics for module registration.
        """
        if self.metrics_collector:
            try:
                timestamp = time.time()
                await self.metrics_collector.add_metric_point(
                    "module_registrations",
                    1.0,
                    timestamp,
                    {"module_id": module_id, "class": module_class.__name__}
                )
                await self.metrics_collector.add_metric_point(
                    f"module_{module_id}_status",
                    1.0,
                    timestamp,
                    {"status": "registered"}
                )
            except (ValueError, TypeError, asyncio.TimeoutError) as e:
                logger.warning("Failed to add module registration metric: %s", e)
            except RuntimeError as e:
                logger.warning(f"Runtime error when adding module registration metric: {e}")

    async def track_status_change(self, module_id: str, status: str):
        """
        Tracks metrics for module status changes.
        """
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
                await self.metrics_collector.add_metric_point(
                    f"module_{module_id}_status",
                    status_code,
                    timestamp,
                    {"status": status}
                )
            except (ValueError, TypeError, asyncio.TimeoutError, AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to add module status metric: {e}")

    async def track_shutdown(self):
        """
        Tracks metrics for module registry shutdown.
        """
        if self.metrics_collector:
            try:
                timestamp = time.time()
                await self.metrics_collector.add_metric_point(
                    "registry_shutdown",
                    1.0,
                    timestamp,
                    {"modules_count": len(self.modules)}
                )
            except (ValueError, TypeError, asyncio.TimeoutError, AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to add shutdown metric: {e}")

class CallbackManager:
    """Manages event callbacks for module registry"""
    def __init__(self):
        self.callbacks: Dict[str, Set[Callable]] = {
            "module_added": set(),
            "module_removed": set(),
            "initialization_complete": set(),
            "error": set(),
            "module_status_change": set()
        }

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Registers a callback for a specific event type.
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)

    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """
        Unregisters a callback for a specific event type.
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].discard(callback)

    async def notify_callbacks(self, event_type: str, data: Any) -> None:
        """
        Notifies all registered callbacks for a specific event type.
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    logger.error("Error in registry callback: %s", e)

class ModuleRegistry:
    """
    Core registry for managing modules in the vAIn P2P AGI system.
    Handles module lifecycle, dependency resolution, and configuration.
    """
    def __init__(self, 
                 config_path: str = "config/modules.json",
                 resource_manager: Optional[ResourceManager] = None,
                 metrics_collector: Any = None):
        self.modules: Dict[str, Any] = {}
        self.module_classes: Dict[str, Type] = {}
        self.module_instances: Dict[str, Any] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.module_status: Dict[str, str] = {}
        self.lock = threading.RLock()
        
        # Initialize supporting components
        self.config_manager = ConfigManager(config_path)
        self.callback_manager = CallbackManager()
        self.metrics_tracker = MetricsTracker(self.modules, resource_manager, metrics_collector)
        self.resource_manager = resource_manager
        
        self.lifecycle_manager = None  # Will be initialized after dependency resolution
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        self._background_task = None
        
        logger.info("Module Registry initialized")
        
    async def initialize(self):
        """
        Initializes the module registry and all registered modules.
        """
        if self._initialized:
            logger.warning("Module Registry already initialized")
            return
            
        logger.info("Initializing Module Registry...")
        try:
            # Load configuration
            config = await self.config_manager.load_configuration()
            if config and "modules" in config:
                for module_id, module_config in config.get("modules", {}).items():
                    if module_id in self.modules:
                        # Update existing module configuration
                        self.modules[module_id].update(module_config)
                    
            # Resolve dependencies
            resolver = DependencyResolver(self.modules, self.dependencies)
            startup_order = resolver.compute_startup_order()
            
            # Initialize lifecycle manager
            self.lifecycle_manager = LifecycleManager(self.module_instances, startup_order)
            await self.lifecycle_manager.initialize_modules()
            
            # Start periodic status checks
            self._background_task = asyncio.create_task(self._periodic_status_check())
            
            self._initialized = True
            logger.info("Module Registry initialization complete")
            
            # Notify listeners
            await self.callback_manager.notify_callbacks(
                "initialization_complete", 
                {"module_count": len(self.modules)}
            )
            
        except Exception as e:
            logger.error("Error initializing Module Registry: %s", e)
            await self.callback_manager.notify_callbacks(
                "error", 
                {"error": str(e), "component": "ModuleRegistry", "action": "initialize"}
            )
            raise
            
    async def shutdown(self):
        """
        Shuts down the module registry and all registered modules.
        """
        if not self._initialized:
            logger.warning("Module Registry not initialized, nothing to shut down")
            return
            
        logger.info("Shutting down Module Registry...")
        
        # Set shutdown event to stop background tasks
        self._shutdown_event.set()
        
        # Wait for background task to complete
        if self._background_task and not self._background_task.done():
            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Background task did not complete in time, cancelling")
                self._background_task.cancel()
        
        # Save configuration
        await self.config_manager.save_configuration({
            module_id: {"status": status} 
            for module_id, status in self.module_status.items()
        })
        
        # Shut down modules
        if self.lifecycle_manager:
            await self.lifecycle_manager.shutdown_modules()
        
        # Track shutdown in metrics
        await self.metrics_tracker.track_shutdown()
        
        self._initialized = False
        logger.info("Module Registry shutdown complete")
    
    def register_module(self, module_id: str, module_class: Type, dependencies: List[str] = None) -> None:
        """
        Registers a module with the registry.
        
        Args:
            module_id: Unique identifier for the module
            module_class: The class of the module
            dependencies: List of module IDs this module depends on
        """
        with self.lock:
            if module_id in self.module_classes:
                raise ModuleRegistryError(f"Module {module_id} is already registered")
                
            self.module_classes[module_id] = module_class
            self.modules[module_id] = {"id": module_id, "class": module_class.__name__}
            self.dependencies[module_id] = dependencies or []
            self.module_status[module_id] = "registered"
            
            # Instantiate the module
            try:
                instance = module_class()
                self.module_instances[module_id] = instance
            except (TypeError, ValueError, AttributeError, ImportError, RuntimeError) as e:
                logger.error("Failed to instantiate module %s: %s", module_id, e)
                self.module_status[module_id] = "error"
                raise ModuleRegistryError(f"Failed to instantiate module {module_id}: {e}") from e
                
            logger.info("Registered module: %s (%s)", module_id, module_class.__name__)
            
            # Track in metrics
            asyncio.create_task(self.metrics_tracker.track_registration(module_id, module_class))
            
            # Notify listeners
            asyncio.create_task(self.callback_manager.notify_callbacks(
                "module_added",
                {"module_id": module_id, "class": module_class.__name__}
            ))
    
    def unregister_module(self, module_id: str) -> None:
        """
        Unregisters a module from the registry.
        
        Args:
            module_id: ID of the module to unregister
        """
        with self.lock:
            if module_id not in self.modules:
                raise ModuleRegistryError(f"Module {module_id} is not registered")
                
            # Check if other modules depend on this one
            dependents = []
            for dep_id, deps in self.dependencies.items():
                if module_id in deps:
                    dependents.append(dep_id)
                    
            if dependents:
                raise ModuleRegistryError(
                    f"Cannot unregister module {module_id} because it is required by: {', '.join(dependents)}"
                )
                
            # Remove module
            module_info = self.modules.pop(module_id)
            self.module_classes.pop(module_id)
            self.dependencies.pop(module_id)
            self.module_status.pop(module_id)
            instance = self.module_instances.pop(module_id, None)
            
            # Shutdown module if it's running
            if instance and hasattr(instance, 'shutdown'):
                try:
                    shutdown_method = getattr(instance, 'shutdown')
                    if inspect.iscoroutinefunction(shutdown_method):
                        # We can't await here directly, so we should handle it in a cleaner way
                        # For now, just create a task
                        asyncio.create_task(shutdown_method())
                    else:
                        shutdown_method()
                except Exception as e:
                    logger.error("Error shutting down module %s: %s", module_id, e)
            
            logger.info("Unregistered module: %s", module_id)
            
            # Notify listeners
            asyncio.create_task(self.callback_manager.notify_callbacks(
                "module_removed",
                {"module_id": module_id, "class": module_info.get("class", "")}
            ))
    
    def get_module(self, module_id: str) -> Any:
        """
        Gets a module instance by ID.
        
        Args:
            module_id: ID of the module to get
            
        Returns:
            The module instance
        """
        with self.lock:
            if module_id not in self.module_instances:
                raise ModuleRegistryError(f"Module {module_id} is not registered")
            return self.module_instances[module_id]
    
    def update_module_status(self, module_id: str, status: str) -> None:
        """
        Updates the status of a module.
        
        Args:
            module_id: ID of the module
            status: New status (registered, initializing, active, suspended, error, terminated)
        """
        with self.lock:
            if module_id not in self.modules:
                raise ModuleRegistryError(f"Module {module_id} is not registered")
                
            valid_statuses = ["registered", "initializing", "active", "suspended", "error", "terminated"]
            if status not in valid_statuses:
                raise ModuleRegistryError(f"Invalid status: {status}. Must be one of {valid_statuses}")
                
            old_status = self.module_status.get(module_id)
            if old_status != status:
                self.module_status[module_id] = status
                logger.info("Module %s status changed: %s -> %s", module_id, old_status, status)
                
                # Track in metrics
                asyncio.create_task(self.metrics_tracker.track_status_change(module_id, status))
                
                # Notify listeners
                asyncio.create_task(self.callback_manager.notify_callbacks(
                    "module_status_change",
                    {"module_id": module_id, "old_status": old_status, "new_status": status}
                ))
    
    async def _periodic_status_check(self, interval: float = 60.0):
        """
        Periodically checks the status of modules and updates the configuration.
        """
        logger.info("Starting periodic module status checks every %s seconds", interval)
        while not self._shutdown_event.is_set():
            try:
                # Check if any modules need attention
                for module_id, instance in self.module_instances.items():
                    status = self.module_status.get(module_id)
                    
                    # Check if module is responsive if it's active
                    if status == "active" and hasattr(instance, 'heartbeat'):
                        try:
                            heartbeat_method = getattr(instance, 'heartbeat')
                            if inspect.iscoroutinefunction(heartbeat_method):
                                result = await asyncio.wait_for(heartbeat_method(), timeout=5.0)
                            else:
                                result = heartbeat_method()
                                
                            if not result:  # If heartbeat returns False, mark as suspended
                                self.update_module_status(module_id, "suspended")
                        except (asyncio.TimeoutError, TypeError, AttributeError, ValueError) as e:
                            logger.warning("Module %s heartbeat failed: %s", module_id, e)
                            self.update_module_status(module_id, "error")
                        except RuntimeError as e:
                            logger.warning("Runtime error in module %s heartbeat: %s", module_id, e)
                            self.update_module_status(module_id, "error")
                
                # Save current configuration
                await self.config_manager.save_configuration({
                    module_id: {"status": status} 
                    for module_id, status in self.module_status.items()
                })
                
            except (asyncio.TimeoutError, TypeError, KeyError, AttributeError, ValueError) as e:
                logger.error("Error in periodic status check: %s", e)
            except RuntimeError as e:
                logger.error("Runtime error in periodic status check: %s", e)
            
            try:
                # Wait for the next interval or until shutdown
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                # This is expected when the timeout is reached and shutdown_event is not set
                pass