import logging
import asyncio
import signal
import sys
import time
import os
import gc
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_core.module_registry import ModuleRegistry
from .resource_management import ResourceManager
from ai_core.metrics_collector import MetricsCollector, MetricsConfig, MetricPoint
from ai_core.model_storage import ModelStorage
from ai_core.chatbot.module_integration import ModuleIntegration, ModuleIntegrationConfig
from core.interactive_utils import InteractiveSession, InteractiveConfig
from utils.unified_logger import get_logger
from utils.memory_manager import memory_manager
from debugging.debug_config import DebugConfigManager
from debugging.port_manager import DebugPortManager

logger = get_logger("system_coordinator")

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
    recovery_scripts_dir: str = "./recovery_scripts"
    use_unified_logging: bool = True
    log_config_path: Optional[str] = "config/logging.json"
    debug_config_path: Optional[str] = "config/debug.json"
    memory_monitoring: bool = True
    memory_warning_threshold: float = 75.0
    memory_critical_threshold: float = 85.0
    debug_port: int = 5678

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
        
        # Initialize logging using new unified logger
        if self.config.use_unified_logging:
            self.logger = get_logger("system_coordinator", self.config.log_config_path)
        else:
            self._setup_logging()  # Fall back to old method
            
        # Initialize debug configuration manager
        self.debug_config_manager = DebugConfigManager.get_instance()
        if self.config.debug_config_path:
            self.debug_config_manager.config_path = self.config.debug_config_path
            
        # Initialize memory manager if enabled
        if self.config.memory_monitoring:
            memory_manager.set_thresholds(
                warning=self.config.memory_warning_threshold,
                critical=self.config.memory_critical_threshold
            )
            memory_manager.start_monitoring(interval=60, trace_memory=True)
            memory_manager.register_callback("critical", self._handle_critical_memory)
        
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
    
    def _handle_critical_memory(self, stats) -> None:
        """Handle critical memory condition"""
        logger.warning(f"Critical memory condition: {stats.system_percent}% system, {stats.process_mb:.1f}MB process")
        
        # Force garbage collection
        memory_manager.force_garbage_collection()
        
        # Log memory-heavy components for investigation
        if hasattr(stats, 'largest_objects') and stats.largest_objects:
            top_consumers = stats.largest_objects[:3]
            for i, obj in enumerate(top_consumers):
                logger.warning(f"Memory consumer #{i+1}: {obj['file']}:{obj['line']} - {obj['size_mb']:.1f}MB")
                
        # Notify callbacks about memory issue
        self._notify_callbacks('error', {
            'phase': 'runtime',
            'error': f"Critical memory usage: {stats.system_percent}%",
            'timestamp': time.time()
        })

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
            
            # Reserve debug port if needed
            if hasattr(self.config, 'debug_port') and self.config.debug_port:
                port = DebugPortManager.reserve_port(self.config.debug_port)
                if port is None:
                    logger.warning(f"Could not reserve debug port {self.config.debug_port}, finding alternative")
                    port = DebugPortManager.find_available_port()
                    
                if port:
                    self.debug_port = port
                    logger.info(f"Reserved debug port: {port}")
                else:
                    logger.warning("Could not reserve any debug port")

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
            
            # Track system components in memory manager
            if self.config.memory_monitoring:
                if self.module_registry:
                    memory_manager.track_object(self.module_registry, "Module Registry")
                if self.resource_manager:
                    memory_manager.track_object(self.resource_manager, "Resource Manager")
                if self.metrics_collector:
                    memory_manager.track_object(self.metrics_collector, "Metrics Collector")
            
            # Mark initialization as complete
            self.is_initialized = True
            self.startup_complete = True
            initialization_time = time.time() - start_time
            
            logger.info("System initialization complete in %.2f seconds", initialization_time)
            
            # Notify callbacks
            await self._notify_callbacks('startup_complete', {
                'timestamp': time.time(),
                'initialization_time': initialization_time,
                'components': dict(self.component_status)
            })
            
            return True
            
        except (RuntimeError, IOError, ValueError, asyncio.TimeoutError) as e:
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
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, 
                IOError, OSError, asyncio.TimeoutError, LookupError) as e:
            self.initialization_error = f"Unexpected error: {str(e)}"
            logger.critical(f"Critical failure during initialization: {e}", exc_info=True)
            
            # Attempt cleanup of partially initialized components
            await self._cleanup_failed_initialization()
            
            # Notify error callbacks
            await self._notify_callbacks('error', {
                'phase': 'initialization',
                'error': f"Unexpected error: {str(e)}",
                'timestamp': time.time()
            })
            
            return False
        except Exception as e:  # Fallback for truly unexpected errors
            self.initialization_error = f"Critical error: {str(e)}"
            logger.critical(f"Critical failure during initialization: {e}", exc_info=True)
            
            # Attempt cleanup of partially initialized components
            await self._cleanup_failed_initialization()
            
            # Notify error callbacks
            await self._notify_callbacks('error', {
                'phase': 'initialization',
                'error': f"Critical error: {str(e)}",
                'timestamp': time.time()
            })
            
            return False

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
            
            # Stop memory monitoring
            if self.config.memory_monitoring:
                memory_manager.stop_monitoring()
            
            # Shutdown thread executor
            self._thread_executor.shutdown(wait=False)
            
            # Run garbage collection
            gc.collect()
            
            # Calculate shutdown time
            shutdown_time = time.time() - start_time
            logger.info(f"System shutdown {'completed successfully' if shutdown_success else 'completed with errors'} in {shutdown_time:.2f} seconds")
            
        except (RuntimeError, ValueError, TypeError, asyncio.CancelledError, IOError, OSError) as e:
            logger.error(f"Error during system shutdown: {e}")
            shutdown_success = False
        except Exception as e:  # Still catch unexpected errors, but log with traceback
            logger.error(f"Unexpected critical error during system shutdown", exc_info=True)
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
        
        # Add memory stats if monitoring enabled
        if self.config.memory_monitoring:
            try:
                memory_stats = memory_manager.get_memory_stats()
                status["memory"] = {
                    "system_percent": memory_stats.system_percent,
                    "process_mb": memory_stats.process_mb,
                    "available_mb": memory_stats.available_system_mb,
                    "tracked_objects": memory_stats.tracked_objects
                }
            except Exception as e:
                logger.warning(f"Error getting memory stats: {e}")
        
        # Get more detailed status if modules are initialized
        if self.module_registry:
            try:
                registry_status = await self.module_registry.get_system_metrics()
                status["module_registry_status"] = registry_status
            except Exception as e:
                logger.warning("Error getting module registry status: %s", e)
                status["module_registry_error"] = str(e)
        
        if self.module_integration:
            try:
                integration_status = await self.module_integration.get_system_status()
                status["module_integration_status"] = integration_status
            except Exception as e:
                logger.warning("Error getting module integration status: %s", e)
                status["module_integration_error"] = str(e)
        
        if self.initialization_error:
            status["initialization_error"] = self.initialization_error
            
        return status
