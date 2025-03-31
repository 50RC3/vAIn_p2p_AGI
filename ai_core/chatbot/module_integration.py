import logging
import asyncio
import time
import torch
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
import os

from .interface import ChatbotInterface, ChatResponse
from .mobile_interface import MobileChatInterface
from .learning_coordinator import LearningCoordinator, LearningCoordinatorConfig
from .rl_trainer import RLTrainer, RLConfig, TrainerState
from ai_core.model_storage import ModelStorage
from ai_core.resource_management import ResourceManager
from ai_core.metrics_collector import MetricsCollector, MetricsConfig

logger = logging.getLogger(__name__)

@dataclass
class ModuleIntegrationConfig:
    """Configuration for module integration"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resource_monitoring_interval: int = 60  # seconds
    memory_threshold: float = 85.0  # percentage
    enable_distributed: bool = False
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    metrics_collection_interval: int = 30  # seconds
    metrics_storage_path: str = "./logs/metrics"
    max_startup_retries: int = 3
    startup_retry_delay: float = 5.0  # seconds
    shutdown_timeout: int = 30  # seconds

class ModuleIntegration:
    """Integrates all chatbot modules for seamless operation"""
    
    def __init__(self, config: ModuleIntegrationConfig):
        self.config = config
        self._setup_logging()
        self.chatbot_interfaces: Dict[str, ChatbotInterface] = {}
        self.learning_coordinator: Optional[LearningCoordinator] = None
        self.rl_trainer: Optional[RLTrainer] = None
        self.model_storage: Optional[ModelStorage] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.resource_monitor_task = None
        self.is_initialized = False
        self.callbacks: Dict[str, Set[Callable]] = {
            'status_change': set(),
            'resource_warning': set(),
            'task_complete': set(),
            'error': set(),
            'metrics_alert': set(),
            'initialization': set()
        }
        
        # Ensure checkpoint directory exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Set up proper logging for production"""
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
            error_handler = logging.FileHandler("errors.log")
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)
    
    async def initialize(self, 
                       model_storage: ModelStorage,
                       learning_config: Optional[LearningCoordinatorConfig] = None,
                       rl_config: Optional[RLConfig] = None,
                       resource_manager: Optional[ResourceManager] = None) -> bool:
        """Initialize all modules in the proper order"""
        try:
            logger.info("Starting module integration initialization")
            self.model_storage = model_storage
            
            # Notify initialization start
            await self._notify_callbacks('initialization', {
                'status': 'starting',
                'timestamp': time.time()
            })
            
            # Initialize metrics collector first
            metrics_config = MetricsConfig(
                collection_interval=self.config.metrics_collection_interval,
                storage_path=self.config.metrics_storage_path
            )
            self.metrics_collector = MetricsCollector(metrics_config)
            
            # Register metrics alert callback
            self.metrics_collector.register_callback(
                "alert", 
                lambda data: asyncio.create_task(self._handle_metrics_alert(data))
            )
            
            # Start metrics collection
            await self.metrics_collector.start()
            
            # Initialize resource manager if provided
            self.resource_manager = resource_manager
            if not self.resource_manager:
                logger.info("Resource manager not provided, creating a new instance")
                self.resource_manager = ResourceManager()
                await self.resource_manager.initialize(metrics_collector=self.metrics_collector)
            
            # Initialize learning coordinator first
            if learning_config:
                logger.info("Initializing learning coordinator")
                retries = 0
                while retries < self.config.max_startup_retries:
                    try:
                        self.learning_coordinator = LearningCoordinator(learning_config)
                        break
                    except Exception as e:
                        retries += 1
                        logger.warning(f"Failed to initialize learning coordinator (attempt {retries}/{self.config.max_startup_retries}): {e}")
                        if retries >= self.config.max_startup_retries:
                            logger.error("Maximum retries reached for learning coordinator initialization")
                            break
                        await asyncio.sleep(self.config.startup_retry_delay)
            
            # Initialize RL trainer if config provided
            if rl_config and self.learning_coordinator:
                logger.info("Initializing RL trainer")
                retries = 0
                while retries < self.config.max_startup_retries:
                    try:
                        # Placeholder - would create model and initialize RL trainer
                        # self.rl_trainer = RLTrainer(model, rl_config)
                        
                        # Register RL trainer with learning coordinator
                        if self.learning_coordinator and self.rl_trainer:
                            self.learning_coordinator.register_rl_trainer(self.rl_trainer)
                        break
                    except Exception as e:
                        retries += 1
                        logger.warning(f"Failed to initialize RL trainer (attempt {retries}/{self.config.max_startup_retries}): {e}")
                        if retries >= self.config.max_startup_retries:
                            logger.error("Maximum retries reached for RL trainer initialization")
                            break
                        await asyncio.sleep(self.config.startup_retry_delay)
            
            # Start resource monitoring
            self._start_resource_monitoring()
            
            self.is_initialized = True
            
            # Notify initialization complete
            await self._notify_callbacks('initialization', {
                'status': 'complete',
                'timestamp': time.time(),
                'modules': {
                    'metrics_collector': self.metrics_collector is not None,
                    'resource_manager': self.resource_manager is not None,
                    'learning_coordinator': self.learning_coordinator is not None,
                    'rl_trainer': self.rl_trainer is not None
                }
            })
            
            logger.info("Module integration initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize module integration: {e}", exc_info=True)
            await self._notify_callbacks('error', {
                'component': 'module_integration',
                'method': 'initialize',
                'error': str(e)
            })
            
            # Try to clean up any partially initialized components
            await self._cleanup_failed_initialization()
            
            return False
    
    async def _cleanup_failed_initialization(self) -> None:
        """Clean up after a failed initialization"""
        try:
            logger.info("Cleaning up after failed initialization")
            
            # Stop metrics collector if it was started
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            # Shut down resource manager if it was initialized by us
            if self.resource_manager and not self.resource_manager.is_initialized:
                await self.resource_manager.shutdown()
            
            # Cancel any active tasks
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
                try:
                    await self.resource_monitor_task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"Error during initialization cleanup: {e}")

    def register_chatbot_interface(self, 
                                  interface_id: str,
                                  interface: ChatbotInterface) -> None:
        """Register a chatbot interface with the integration layer"""
        try:
            if interface_id in self.chatbot_interfaces:
                logger.warning(f"Overriding existing interface with ID {interface_id}")
                
            self.chatbot_interfaces[interface_id] = interface
            
            # Connect learning coordinator to interface
            if self.learning_coordinator:
                # Register learning progress callback
                self.learning_coordinator.register_callback(
                    'learning_progress',
                    lambda data: asyncio.create_task(self._relay_to_interface(
                        interface_id, 'learning_progress', data
                    ))
                )
                
                # Register model saved callback  
                self.learning_coordinator.register_callback(
                    'model_saved',
                    lambda data: asyncio.create_task(self._relay_to_interface(
                        interface_id, 'model_saved', data
                    ))
                )
            
            # Connect RL trainer to interface if available
            if self.rl_trainer:
                self.rl_trainer.register_callback(
                    'update_completed',
                    lambda data: asyncio.create_task(self._relay_to_interface(
                        interface_id, 'rl_update', data
                    ))
                )
                
            logger.info(f"Registered chatbot interface {interface_id}")
            
        except Exception as e:
            logger.error(f"Failed to register interface {interface_id}: {e}")
    
    async def _relay_to_interface(self, 
                                interface_id: str, 
                                event_type: str, 
                                data: Any) -> None:
        """Relay events to specific interface"""
        try:
            if interface_id in self.chatbot_interfaces:
                interface = self.chatbot_interfaces[interface_id]
                
                # Handle both naming patterns for backward compatibility
                if hasattr(interface, '_notify_handlers'):
                    await interface._notify_handlers(event_type, data)
                elif hasattr(interface, '_trigger_event'):
                    await interface._trigger_event(event_type, data)
                else:
                    logger.warning(f"Interface {interface_id} has no handler method")
        except Exception as e:
            logger.error(f"Failed to relay event to interface {interface_id}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for integration events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify all callbacks for a specific event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
    
    async def process_message(self,
                           interface_id: str,
                           user_id: str,
                           message: str,
                           context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Process a message through the specified interface with full integration"""
        try:
            # Ensure initialization
            if not self.is_initialized:
                raise RuntimeError("Module integration not initialized")
                
            # Get the appropriate interface
            if interface_id not in self.chatbot_interfaces:
                raise ValueError(f"Interface with ID {interface_id} not found")
                
            interface = self.chatbot_interfaces[interface_id]
            
            # Process the message
            start_time = time.time()
            response = await interface.process_message(message, context)
            processing_time = time.time() - start_time
            
            # If learning coordinator exists, process for learning in background
            if self.learning_coordinator:
                # Don't await this to avoid blocking
                asyncio.create_task(self._process_for_learning(message))
            
            # Log performance metrics
            logger.debug(f"Message processing time: {processing_time:.3f}s on interface {interface_id}")
            
            # Record processing metrics if metrics collector is available
            if self.metrics_collector:
                asyncio.create_task(self._record_processing_metrics(
                    interface_id, user_id, len(message), processing_time
                ))
            
            # Notify task completion
            await self._notify_callbacks('task_complete', {
                'interface_id': interface_id,
                'user_id': user_id,
                'processing_time': processing_time
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._notify_callbacks('error', {
                'interface_id': interface_id,
                'user_id': user_id,
                'message': message,
                'error': str(e)
            })
            
            # Return error response
            if 'ChatResponse' in globals():
                return ChatResponse(
                    text="Sorry, there was an error processing your message.",
                    confidence=0.0,
                    model_version="error",
                    latency=0.0,
                    error=str(e)
                )
            else:
                # Simple error dict if ChatResponse isn't available
                return {
                    "text": "Sorry, there was an error processing your message.",
                    "error": str(e)
                }
    
    async def _record_processing_metrics(self, interface_id: str, user_id: str, 
                                       message_length: int, processing_time: float) -> None:
        """Record metrics for message processing"""
        try:
            if self.metrics_collector:
                timestamp = time.time()
                
                # Record processing time
                await self.metrics_collector._add_metric_point(
                    f"processing_time_{interface_id}",
                    processing_time,
                    timestamp,
                    {"user_id": user_id, "message_length": message_length}
                )
                
                # Record message length
                await self.metrics_collector._add_metric_point(
                    f"message_length_{interface_id}",
                    float(message_length),
                    timestamp,
                    {"user_id": user_id}
                )
        except Exception as e:
            logger.warning(f"Failed to record processing metrics: {e}")
    
    async def _process_for_learning(self, message: str) -> None:
        """Process message for continuous learning"""
        try:
            if self.learning_coordinator:
                await self.learning_coordinator.process_text(message)
        except Exception as e:
            logger.error(f"Error processing for learning: {e}")
    
    def _start_resource_monitoring(self) -> None:
        """Start monitoring system resources"""
        self.resource_monitor_task = asyncio.create_task(
            self._monitor_resources_periodically()
        )
    
    async def _monitor_resources_periodically(self) -> None:
        """Monitor system resources periodically"""
        try:
            while True:
                await asyncio.sleep(self.config.resource_monitoring_interval)
                await self._check_resources()
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
    
    async def _check_resources(self) -> None:
        """Check system resources and take action if needed"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > self.config.memory_threshold:
                logger.warning(f"Memory usage high: {memory.percent}%")
                await self._notify_callbacks('resource_warning', {
                    'resource': 'memory',
                    'usage': memory.percent,
                    'threshold': self.config.memory_threshold
                })
                
                # Take action - clean up cache
                await self._cleanup_resources()
            
            # Check CPU (optional)
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90:
                logger.warning(f"CPU usage high: {cpu_percent}%")
                await self._notify_callbacks('resource_warning', {
                    'resource': 'cpu',
                    'usage': cpu_percent,
                    'threshold': 90
                })
            
            # Check GPU if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        # Calculate percentage of allocated memory relative to total memory
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        gpu_percent = (allocated / total) * 100
                        
                        if gpu_percent > self.config.memory_threshold:
                            logger.warning(f"GPU {i} memory usage high: {gpu_percent:.1f}%")
                            await self._notify_callbacks('resource_warning', {
                                'resource': f'gpu_{i}',
                                'usage': gpu_percent,
                                'threshold': self.config.memory_threshold
                            })
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Error checking GPU {i} memory: {e}")
                
        except ImportError:
            logger.warning("psutil not available, skipping resource check")
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
    
    async def _handle_metrics_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle alerts from metrics collector"""
        try:
            logger.warning(f"Metrics alert received: {alert_data}")
            
            # Take action based on alert type
            if alert_data.get("metric") == "memory_percent" or "gpu" in alert_data.get("metric", ""):
                # Clean up resources
                await self._cleanup_resources()
            
            # Forward alert to callbacks
            await self._notify_callbacks("metrics_alert", alert_data)
        except Exception as e:
            logger.error(f"Error handling metrics alert: {e}")
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources when memory usage is high"""
        try:
            # Clear caches for all interfaces
            for interface_id, interface in self.chatbot_interfaces.items():
                if hasattr(interface, 'response_cache'):
                    interface.response_cache.clear()
                if hasattr(interface, 'context_cache'):
                    interface.context_cache.clear()
                if hasattr(interface, 'feedback_cache'):
                    interface.feedback_cache.clear()
            
            # Clear torch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Cleaned up resource caches")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics from all modules"""
        stats = {}
        
        try:
            # Get learning coordinator stats
            if self.learning_coordinator:
                stats['learning'] = {
                    'total_examples': self.learning_coordinator.stats.total_examples,
                    'self_supervised': self.learning_coordinator.stats.self_supervised,
                    'unsupervised': self.learning_coordinator.stats.unsupervised,
                    'reinforcement': self.learning_coordinator.stats.reinforcement
                }
            
            # Get RL trainer stats
            if self.rl_trainer:
                stats['rl_training'] = self.rl_trainer.get_training_stats()
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {'error': str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "timestamp": time.time(),
            "initialized": self.is_initialized,
            "interfaces": list(self.chatbot_interfaces.keys()),
            "learning_coordinator": self.learning_coordinator is not None,
            "rl_trainer": self.rl_trainer is not None
        }
        
        # Add metrics if available
        if self.metrics_collector:
            status["metrics"] = await self.metrics_collector.get_current_metrics()
        
        # Add resource manager status
        if self.resource_manager:
            status["registered_modules"] = len(self.resource_manager.registered_modules)
            
            # Add status of registered modules
            modules_status = {}
            for module_id, module_info in self.resource_manager.registered_modules.items():
                modules_status[module_id] = {
                    "status": module_info["status"],
                    "importance": module_info["importance"],
                    "last_active": time.time() - module_info["last_active"]  # seconds since last active
                }
            status["modules"] = modules_status
            
        return status
    
    async def shutdown(self) -> None:
        """Gracefully shut down all modules"""
        logger.info("Shutting down module integration")
        
        shutdown_success = True
        
        # Cancel resource monitoring
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await asyncio.wait_for(self.resource_monitor_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Clean up learning coordinator (with timeout)
        if self.learning_coordinator:
            try:
                await asyncio.wait_for(
                    self.learning_coordinator.cleanup(),
                    timeout=self.config.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.error("Learning coordinator cleanup timed out")
                shutdown_success = False
            except Exception as e:
                logger.error(f"Error shutting down learning coordinator: {e}")
                shutdown_success = False
        
        # Clean up interfaces (with timeout)
        interface_tasks = []
        for interface_id, interface in self.chatbot_interfaces.items():
            try:
                if hasattr(interface, 'clear_session'):
                    interface_tasks.append(interface.clear_session())
            except Exception as e:
                logger.error(f"Error preparing interface {interface_id} for shutdown: {e}")
        
        if interface_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*interface_tasks, return_exceptions=True),
                    timeout=self.config.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.error("Interface cleanup timed out")
                shutdown_success = False
        
        # Clean up metrics collector (with timeout)
        if self.metrics_collector:
            try:
                await asyncio.wait_for(
                    self.metrics_collector.stop(),
                    timeout=self.config.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.error("Metrics collector shutdown timed out")
                shutdown_success = False
            except Exception as e:
                logger.error(f"Error shutting down metrics collector: {e}")
                shutdown_success = False
        
        # Final resource cleanup
        try:
            await self._cleanup_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            shutdown_success = False
        
        # Shutdown resource manager (with timeout)
        if self.resource_manager:
            try:
                await asyncio.wait_for(
                    self.resource_manager.shutdown(),
                    timeout=self.config.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.error("Resource manager shutdown timed out")
                shutdown_success = False
            except Exception as e:
                logger.error(f"Error shutting down resource manager: {e}")
                shutdown_success = False
        
        logger.info(f"Module integration shutdown {'completed successfully' if shutdown_success else 'completed with errors'}")
