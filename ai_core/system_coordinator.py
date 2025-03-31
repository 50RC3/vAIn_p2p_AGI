import logging
import asyncio
import signal
import sys
import time
import os
import gc
from typing import Dict, Any, Optional, List, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from ai_core.module_registry import ModuleRegistry
from ai_core.resource_management import ResourceManager
from ai_core.metrics_collector import MetricsCollector, MetricsConfig
from ai_core.model_storage import ModelStorage
from ai_core.chatbot.module_integration import ModuleIntegration, ModuleIntegrationConfig
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

@dataclass
class SystemCoordinatorConfig:
    """Configuration for the system coordinator"""
    interactive: bool = True
    startup_timeout: float = 60.0  # seconds
    shutdown_timeout: float = 60.0  # seconds
    metrics_enabled: bool = True
    metrics_storage_path: str = "./logs/metrics"
    log_level: str = "INFO"
    checkpoint_dir: str = "./checkpoints"
    resource_monitor_interval: int = 60  # seconds
    memory_threshold: float = 85.0  # percentage
    enable_distributed: bool = False
    max_startup_retries: int = 3
    startup_retry_delay: float = 5.0  # seconds
    module_load_timeout: float = 30.0  # seconds
    verify_component_health: bool = True
    watchdog_interval: int = 120  # seconds
    auto_restart_failed: bool = True
    backup_interval: int = 3600  # seconds (1 hour)
    thread_pool_size: int = 4

class SystemCoordinator:
    """
    Coordinates the startup, operation, and shutdown of all system modules.
    
    This class provides:
    - Orderly startup and shutdown sequences
    - Global resource management
    - Centralized metrics collection
    - Exception handling and recovery
    - Signal handling for graceful termination
    - System health monitoring and watchdog
    - Automatic component recovery
    """
    
    def __init__(self, config: Optional[SystemCoordinatorConfig] = None):
        self.config = config or SystemCoordinatorConfig()
        self._setup_logging()
        
        # Core components
        self.module_registry = None
        self.resource_manager = None
        self.metrics_collector = None
        self.model_storage = None
        self.module_integration = None
        self.interactive_session = None
        
        # State tracking
        self.is_initialized = False
        self.is_shutting_down = False
        self.startup_complete = False
        self.shutdown_complete = False
        self._original_signal_handlers = {}
        self.initialization_error = None
        self.component_status = {}
        
        # Task management
        self._watchdog_task = None
        self._backup_task = None
        self._active_tasks = set()
        self._thread_executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Callbacks
        self.callbacks = {
            'startup_complete': set(),
            'shutdown_started': set(),
            'shutdown_complete': set(),
            'error': set(),
            'component_failure': set(),
            'component_recovered': set(),
            'system_health': set(),
        }
    
    def _setup_logging(self) -> None:
        """Configure logging for the system coordinator"""
        log_level = getattr(logging, self.config.log_level, logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Create handlers if they don't exist
        if not root_logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            
            # File handler for errors
            os.makedirs("logs", exist_ok=True)
            error_handler = logging.FileHandler("logs/errors.log")
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)
            
            # File handler for all logs
            all_handler = logging.FileHandler("logs/system.log")
            all_handler.setLevel(log_level)
            all_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            all_handler.setFormatter(all_formatter)
            root_logger.addHandler(all_handler)
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful termination"""
        try:
            # Save original handlers
            if hasattr(signal, 'SIGINT'):
                self._original_signal_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handle_interrupt)
            
            if hasattr(signal, 'SIGTERM'):
                self._original_signal_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, self._handle_termination)
                
            logger.debug("Signal handlers set up")
        except Exception as e:
            logger.warning(f"Failed to set up signal handlers: {e}")
    
    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers"""
        try:
            for sig, handler in self._original_signal_handlers.items():
                signal.signal(sig, handler)
            logger.debug("Original signal handlers restored")
        except Exception as e:
            logger.warning(f"Failed to restore signal handlers: {e}")
    
    def _handle_interrupt(self, signum, frame) -> None:
        """Handle SIGINT (Ctrl+C)"""
        if not self.is_shutting_down:
            logger.info("Interrupt signal received, initiating shutdown...")
            self.is_shutting_down = True
            
            # Create a new event loop for the shutdown process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self.shutdown())
            except Exception as e:
                logger.error(f"Error during emergency shutdown: {e}")
            finally:
                loop.close()
                sys.exit(1)
    
    def _handle_termination(self, signum, frame) -> None:
        """Handle SIGTERM"""
        if not self.is_shutting_down:
            logger.info("Termination signal received, initiating shutdown...")
            self.is_shutting_down = True
            
            # Create a new event loop for the shutdown process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self.shutdown())
            except Exception as e:
                logger.error(f"Error during emergency shutdown: {e}")
            finally:
                loop.close()
                sys.exit(0)
    
    async def initialize(self) -> bool:
        """
        Initialize all system components in the correct order.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.is_initialized:
            logger.warning("System already initialized")
            return True
            
        try:
            logger.info("Starting system initialization")
            start_time = time.time()
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Set up interactive session if enabled
            if self.config.interactive:
                interactive_config = InteractiveConfig(
                    timeout=30.0,
                    persistent_state=True,
                    safe_mode=True,
                    progress_tracking=True,
                    cleanup_timeout=self.config.shutdown_timeout
                )
                self.interactive_session = InteractiveSession("system_coordinator", config=interactive_config)
                await self.interactive_session.initialize()
            
            # Initialize metrics collector first
            if self.config.metrics_enabled:
                logger.info("Initializing metrics collector")
                metrics_config = MetricsConfig(
                    collection_interval=30,
                    storage_path=self.config.metrics_storage_path
                )
                self.metrics_collector = MetricsCollector(metrics_config)
                await self.metrics_collector.start()
                self.component_status["metrics_collector"] = "active"
            
            # Initialize resource manager
            logger.info("Initializing resource manager")
            self.resource_manager = ResourceManager()
            success = await self.resource_manager.initialize(metrics_collector=self.metrics_collector)
            if not success:
                raise RuntimeError("Failed to initialize resource manager")
            self.component_status["resource_manager"] = "active"
            
            # Initialize module registry
            logger.info("Initializing module registry")
            self.module_registry = ModuleRegistry.get_instance()
            success = await self.module_registry.initialize(
                resource_manager=self.resource_manager,
                metrics_collector=self.metrics_collector
            )
            if not success:
                raise RuntimeError("Failed to initialize module registry")
            self.component_status["module_registry"] = "active"
            
            # Initialize model storage
            logger.info("Initializing model storage")
            self.model_storage = ModelStorage(storage_dir="./model_storage")
            self.component_status["model_storage"] = "active"
            
            # Initialize module integration
            logger.info("Initializing module integration")
            module_integration_config = ModuleIntegrationConfig(
                device="cuda" if self.config.enable_distributed else "cpu",
                resource_monitoring_interval=self.config.resource_monitor_interval,
                memory_threshold=self.config.memory_threshold,
                enable_distributed=self.config.enable_distributed,
                checkpoint_dir=self.config.checkpoint_dir,
                log_level=self.config.log_level,
                metrics_collection_interval=30,
                metrics_storage_path=self.config.metrics_storage_path,
                max_startup_retries=self.config.max_startup_retries,
                startup_retry_delay=self.config.startup_retry_delay,
                shutdown_timeout=self.config.shutdown_timeout
            )
            
            self.module_integration = ModuleIntegration(module_integration_config)
            success = await self.module_integration.initialize(
                model_storage=self.model_storage,
                resource_manager=self.resource_manager
            )
            if not success:
                raise RuntimeError("Failed to initialize module integration")
            self.component_status["module_integration"] = "active"
            
            # Start system health monitoring
            if self.config.verify_component_health:
                self._watchdog_task = asyncio.create_task(self._run_system_watchdog())
                self._backup_task = asyncio.create_task(self._run_periodic_backups())
            
            # Mark initialization as complete
            self.is_initialized = True
            self.startup_complete = True
            initialization_time = time.time() - start_time
            
            logger.info(f"System initialization complete in {initialization_time:.2f} seconds")
            
            # Notify callbacks
            await self._notify_callbacks('startup_complete', {
                'timestamp': time.time(),
                'initialization_time': initialization_time,
                'components': dict(self.component_status)
            })
            
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Failed to initialize system: {e}", exc_info=True)
            
            # Attempt cleanup of partially initialized components
            await self._cleanup_failed_initialization()
            
            # Notify error callbacks
            await self._notify_callbacks('error', {
                'phase': 'initialization',
                'error': str(e),
                'timestamp': time.time()
            })
            
            return False
    
    async def _cleanup_failed_initialization(self) -> None:
        """Clean up after a failed initialization"""
        logger.info("Cleaning up after failed initialization")
        
        components = [
            ("module_integration", self.module_integration),
            ("module_registry", self.module_registry),
            ("resource_manager", self.resource_manager),
            ("metrics_collector", self.metrics_collector),
            ("interactive_session", self.interactive_session)
        ]
        
        for name, component in reversed(components):
            if component:
                try:
                    logger.info(f"Cleaning up {name}")
                    if hasattr(component, 'shutdown'):
                        await asyncio.wait_for(
                            component.shutdown(),
                            timeout=self.config.shutdown_timeout
                        )
                        logger.info(f"Successfully cleaned up {name}")
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(f"Error shutting down {name} during cleanup: {e}")
        
        # Restore signal handlers
        self._restore_signal_handlers()
        
        # Clear active tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        
        # Shutdown thread executor
        self._thread_executor.shutdown(wait=False)
        
        # Run garbage collection
        gc.collect()
    
    async def _run_system_watchdog(self) -> None:
        """Monitor system health and recover failed components if possible"""
        try:
            logger.info("Starting system health watchdog")
            
            while not self.is_shutting_down:
                try:
                    # Check component health
                    health_status = await self._check_component_health()
                    
                    # Try to recover failed components
                    if self.config.auto_restart_failed:
                        await self._recover_failed_components(health_status)
                    
                    # Notify health status
                    await self._notify_callbacks('system_health', {
                        'timestamp': time.time(),
                        'status': health_status,
                        'overall_health': all(
                            status == "active" for component, status in health_status.items()
                        )
                    })
                    
                except Exception as e:
                    logger.error(f"Error in watchdog monitoring: {e}")
                
                # Wait for next check
                await asyncio.sleep(self.config.watchdog_interval)
                
        except asyncio.CancelledError:
            logger.info("Watchdog task cancelled")
        except Exception as e:
            logger.error(f"Watchdog task failed: {e}")
    
    async def _check_component_health(self) -> Dict[str, str]:
        """Check health of all system components"""
        health = {}
        
        # Check metrics collector
        if self.metrics_collector:
            try:
                # A simple check if the collector is running
                if getattr(self.metrics_collector, '_is_running', False):
                    health['metrics_collector'] = 'active'
                else:
                    health['metrics_collector'] = 'inactive'
            except Exception as e:
                logger.warning(f"Error checking metrics collector health: {e}")
                health['metrics_collector'] = 'error'
        
        # Check resource manager
        if self.resource_manager:
            try:
                # Check if initialized
                if getattr(self.resource_manager, 'is_initialized', False):
                    health['resource_manager'] = 'active'
                else:
                    health['resource_manager'] = 'inactive'
            except Exception as e:
                logger.warning(f"Error checking resource manager health: {e}")
                health['resource_manager'] = 'error'
        
        # Check module registry
        if self.module_registry:
            try:
                # Try to get metrics as a health check
                await self.module_registry.get_system_metrics()
                health['module_registry'] = 'active'
            except Exception as e:
                logger.warning(f"Error checking module registry health: {e}")
                health['module_registry'] = 'error'
        
        # Check module integration
        if self.module_integration:
            try:
                # Check if it's initialized
                if getattr(self.module_integration, 'is_initialized', False):
                    await self.module_integration.get_system_status()
                    health['module_integration'] = 'active'
                else:
                    health['module_integration'] = 'inactive'
            except Exception as e:
                logger.warning(f"Error checking module integration health: {e}")
                health['module_integration'] = 'error'
        
        # Compare with previous statuses and log changes
        for component, status in health.items():
            previous = self.component_status.get(component)
            if previous and previous != status:
                if status == 'error' or status == 'inactive':
                    logger.warning(f"Component {component} changed state from {previous} to {status}")
                    await self._notify_callbacks('component_failure', {
                        'component': component,
                        'previous_status': previous,
                        'current_status': status,
                        'timestamp': time.time()
                    })
                elif previous == 'error' or previous == 'inactive':
                    logger.info(f"Component {component} recovered from {previous} to {status}")
                    await self._notify_callbacks('component_recovered', {
                        'component': component,
                        'previous_status': previous,
                        'current_status': status,
                        'timestamp': time.time()
                    })
        
        # Update component status
        self.component_status.update(health)
        return health
    
    async def _recover_failed_components(self, health_status: Dict[str, str]) -> None:
        """Attempt to recover failed components"""
        for component, status in health_status.items():
            if status in ('error', 'inactive'):
                logger.info(f"Attempting to recover {component}")
                
                try:
                    if component == 'metrics_collector' and self.metrics_collector:
                        # Try to restart metrics collector
                        await self.metrics_collector.stop()
                        await self.metrics_collector.start()
                        logger.info(f"Successfully restarted metrics collector")
                    
                    elif component == 'resource_manager' and self.resource_manager:
                        # Try to re-initialize resource manager
                        await self.resource_manager.initialize(metrics_collector=self.metrics_collector)
                        logger.info(f"Successfully restarted resource manager")
                    
                    # Note: Module registry and module integration may be more complex to restart
                    # and might require more careful handling
                
                except Exception as e:
                    logger.error(f"Failed to recover {component}: {e}")
    
    async def _run_periodic_backups(self) -> None:
        """Periodically back up system state"""
        try:
            while not self.is_shutting_down:
                try:
                    await self._create_system_backup()
                except Exception as e:
                    logger.error(f"Error creating system backup: {e}")
                
                # Wait for next backup
                await asyncio.sleep(self.config.backup_interval)
                
        except asyncio.CancelledError:
            logger.info("Backup task cancelled")
        except Exception as e:
            logger.error(f"Backup task failed: {e}")
    
    async def _create_system_backup(self) -> None:
        """Create a backup of system state"""
        try:
            timestamp = int(time.time())
            backup_dir = os.path.join(self.config.checkpoint_dir, f"system_backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            
            logger.info(f"Creating system backup at {backup_dir}")
            
            # Get resource manager to create backups for registered modules
            if self.resource_manager and hasattr(self.resource_manager, 'registered_modules'):
                for module_id in self.resource_manager.registered_modules:
                    try:
                        # Use resource manager's backup functionality
                        if hasattr(self.resource_manager, 'create_backup'):
                            # In a real implementation, we would need to get module state
                            # For now we just create an empty placeholder
                            placeholder_data = {"timestamp": timestamp, "module_id": module_id}
                            backup_path = await self.resource_manager.create_backup(module_id, placeholder_data)
                            logger.debug(f"Created backup for {module_id} at {backup_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create backup for {module_id}: {e}")
            
            # Save system status
            status = await self.get_system_status()
            try:
                with open(os.path.join(backup_dir, "system_status.json"), "w") as f:
                    import json
                    json.dump(status, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save system status: {e}")
                
            logger.info(f"System backup completed")
            
        except Exception as e:
            logger.error(f"Error creating system backup: {e}")
    
    async def shutdown(self) -> bool:
        """
        Gracefully shut down all system components in reverse order.
        
        Returns:
            bool: True if shutdown was successful
        """
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return False
        
        if self.shutdown_complete:
            logger.warning("System already shut down")
            return True
            
        self.is_shutting_down = True
        start_time = time.time()
        shutdown_success = True
        
        logger.info("Starting system shutdown")
        
        # Notify shutdown started
        await self._notify_callbacks('shutdown_started', {
            'timestamp': time.time()
        })
        
        try:
            # Cancel monitoring tasks first
            tasks_to_cancel = []
            if self._watchdog_task and not self._watchdog_task.done():
                tasks_to_cancel.append(self._watchdog_task)
            
            if self._backup_task and not self._backup_task.done():
                tasks_to_cancel.append(self._backup_task)
                
            for task in tasks_to_cancel:
                task.cancel()
                
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some monitoring tasks did not cancel cleanly")
            
            # Shut down components in reverse order
            components = [
                ("module_integration", self.module_integration),
                ("module_registry", self.module_registry),
                ("resource_manager", self.resource_manager),
                ("metrics_collector", self.metrics_collector),
                ("interactive_session", self.interactive_session)
            ]
            
            for name, component in components:
                if component:
                    try:
                        logger.info(f"Shutting down {name}")
                        if hasattr(component, 'shutdown'):
                            await asyncio.wait_for(
                                component.shutdown(),
                                timeout=self.config.shutdown_timeout
                            )
                            logger.info(f"Successfully shut down {name}")
                    except asyncio.TimeoutError:
                        logger.error(f"{name} shutdown timed out")
                        shutdown_success = False
                    except Exception as e:
                        logger.error(f"Error shutting down {name}: {e}")
                        shutdown_success = False
            
            # Cancel any remaining active tasks
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()
            
            if self._active_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._active_tasks, return_exceptions=True),
                        timeout=self.config.shutdown_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some active tasks did not complete during shutdown")
            
            # Shutdown thread executor
            self._thread_executor.shutdown(wait=False)
            
            # Run garbage collection
            gc.collect()
            
            # Calculate shutdown time
            shutdown_time = time.time() - start_time
            logger.info(f"System shutdown {'completed successfully' if shutdown_success else 'completed with errors'} in {shutdown_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Unexpected error during system shutdown: {e}")
            shutdown_success = False
        
        finally:
            # Always restore signal handlers
            self._restore_signal_handlers()
            
            # Mark shutdown as complete
            self.shutdown_complete = True
            
            # Notify shutdown complete
            await self._notify_callbacks('shutdown_complete', {
                'timestamp': time.time(),
                'success': shutdown_success,
                'shutdown_time': time.time() - start_time
            })
        
        return shutdown_success
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for a specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify registered callbacks of an event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in {event_type} callback: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "timestamp": time.time(),
            "initialized": self.is_initialized,
            "startup_complete": self.startup_complete,
            "is_shutting_down": self.is_shutting_down,
            "shutdown_complete": self.shutdown_complete,
            "components": {
                "module_registry": self.module_registry is not None,
                "resource_manager": self.resource_manager is not None,
                "metrics_collector": self.metrics_collector is not None,
                "model_storage": self.model_storage is not None,
                "module_integration": self.module_integration is not None,
                "interactive_session": self.interactive_session is not None
            },
            "component_health": dict(self.component_status)
        }
        
        # Get more detailed status if modules are initialized
        if self.module_registry:
            try:
                registry_status = await self.module_registry.get_system_metrics()
                status["module_registry_status"] = registry_status
            except Exception as e:
                logger.warning(f"Error getting module registry status: {e}")
                status["module_registry_error"] = str(e)
        
        if self.module_integration:
            try:
                integration_status = await self.module_integration.get_system_status()
                status["module_integration_status"] = integration_status
            except Exception as e:
                logger.warning(f"Error getting module integration status: {e}")
                status["module_integration_error"] = str(e)
        
        if self.initialization_error:
            status["initialization_error"] = self.initialization_error
            
        return status
    
    async def execute_in_thread(self, func, *args, **kwargs) -> Any:
        """Execute a blocking function in a thread pool to avoid blocking the event loop"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._thread_executor, 
            lambda: func(*args, **kwargs)
        )
    
    def add_task(self, coroutine) -> asyncio.Task:
        """Add a task to be tracked by the coordinator"""
        task = asyncio.create_task(coroutine)
        self._active_tasks.add(task)
        task.add_done_callback(lambda t: self._active_tasks.remove(t))
        return task
    
    async def verify_compatibility(self) -> Tuple[bool, List[str]]:
        """Verify system compatibility and dependencies"""
        issues = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 7):
            issues.append(f"Python version {sys.version} is not supported. Minimum required version is 3.7")
        
        # Check critical dependencies
        try:
            import torch
            if not torch.cuda.is_available() and self.config.enable_distributed:
                issues.append("CUDA is not available but distributed mode is enabled")
        except ImportError:
            issues.append("PyTorch is not installed")
        
        try:
            import aiohttp
        except ImportError:
            issues.append("aiohttp is not installed")
        
        try:
            import psutil
        except ImportError:
            issues.append("psutil is not installed for resource monitoring")
        
        # Check directories are writable
        directories = [
            self.config.checkpoint_dir,
            self.config.metrics_storage_path,
            "logs"
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                test_file = os.path.join(directory, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except (IOError, OSError, PermissionError) as e:
                issues.append(f"Directory {directory} is not writable: {e}")
        
        return len(issues) == 0, issues
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics from all components"""
        metrics = {
            "timestamp": time.time(),
            "system_uptime": time.time() - self.startup_complete if self.startup_complete else 0,
        }
        
        # Get metrics from metrics collector
        if self.metrics_collector:
            try:
                current_metrics = await self.metrics_collector.get_current_metrics()
                metrics["system_metrics"] = current_metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics from metrics collector: {e}")
        
        # Get module metrics from registry
        if self.module_registry:
            try:
                registry_metrics = await self.module_registry.get_system_metrics()
                metrics["module_metrics"] = registry_metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics from module registry: {e}")
        
        # Get learning metrics if available
        if self.module_integration and hasattr(self.module_integration, "get_learning_stats"):
            try:
                learning_stats = await self.module_integration.get_learning_stats()
                metrics["learning_metrics"] = learning_stats
            except Exception as e:
                logger.warning(f"Failed to get learning stats: {e}")
        
        return metrics

async def initialize_system() -> SystemCoordinator:
    """Initialize system with default configuration"""
    config = SystemCoordinatorConfig()
    coordinator = SystemCoordinator(config)
    
    # Check compatibility first
    compatibility, issues = await coordinator.verify_compatibility()
    if not compatibility:
        for issue in issues:
            logger.error(f"Compatibility issue: {issue}")
        raise RuntimeError("System compatibility check failed")
    
    # Initialize the coordinator
    success = await coordinator.initialize()
    if not success:
        raise RuntimeError(f"System initialization failed: {coordinator.initialization_error}")
    
    return coordinator

# Example usage
if __name__ == "__main__":
    async def main():
        try:
            coordinator = await initialize_system()
            logger.info("System running. Press Ctrl+C to shutdown.")
            
            # Keep system running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            if coordinator:
                await coordinator.shutdown()
        except Exception as e:
            logger.error(f"System error: {e}")
            if coordinator:
                await coordinator.shutdown()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
