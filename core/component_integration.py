"""
Component Integration module for vAIn P2P AGI system.
Provides integration utilities for system components.
"""
import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List, Set, Callable, Union, Type
import traceback

from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.interactive_config import InteractionLevel
from training.model_interface import ModelInterface
from ai_core.ui.cli_interface import CommandLineInterface

logger = logging.getLogger(__name__)

class ComponentIntegration:
    """
    Integrates components of the vAIn P2P AGI system to ensure
    proper communication, event handling, and resource management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the component integration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.components: Dict[str, Any] = {}
        self.event_handlers: Dict[str, Set[Callable]] = {}
        self.interactive_session: Optional[InteractiveSession] = None
        self.resource_monitor = None
        self.model_interface: Optional[ModelInterface] = None
        self.cli_interface: Optional[CommandLineInterface] = None
        self.initialized = False
        self.status_tasks = set()
        
    async def initialize(self) -> bool:
        """
        Initialize the component integration.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing component integration")
            
            # Create interactive session if configured
            if self.config.get('interactive_mode', True):
                try:
                    session_id = f"integration_{int(time.time())}"
                    self.interactive_session = InteractiveSession(
                        session_id=session_id,
                        config=InteractiveConfig(
                            timeout=self.config.get('interactive_timeout', 300),
                            max_retries=self.config.get('max_retries', 3),
                            safe_mode=self.config.get('safe_mode', True),
                            memory_threshold=self.config.get('memory_threshold', 0.9),
                            cleanup_timeout=self.config.get('cleanup_timeout', 30),
                            log_interactions=self.config.get('log_interactions', True)
                        )
                    )
                    await self.interactive_session.initialize()
                    logger.info(f"Interactive session {session_id} initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize interactive session: {str(e)}")
            
            # Initialize resource monitoring if configured
            if self.config.get('resource_monitoring', True):
                try:
                    from utils.resource_monitor import default_monitor
                    self.resource_monitor = default_monitor
                    await self.resource_monitor.start_monitoring()
                    logger.info("Resource monitoring initialized")
                except ImportError:
                    logger.warning("Resource monitoring not available")
                except Exception as e:
                    logger.warning(f"Failed to initialize resource monitoring: {str(e)}")
            
            self.initialized = True
            logger.info("Component integration initialized successfully")
            
            # Start periodic status check if configured
            if self.config.get('status_check_interval', 0) > 0:
                task = asyncio.create_task(self._periodic_status_check())
                self.status_tasks.add(task)
                task.add_done_callback(self.status_tasks.discard)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize component integration: {str(e)}")
            return False
    
    async def register_component(self, component_id: str, component: Any) -> bool:
        """
        Register a component with the integration layer.
        
        Args:
            component_id: Unique identifier for the component
            component: The component instance
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            logger.info(f"Registering component: {component_id}")
            
            # Store component
            self.components[component_id] = component
            
            # Special handling for specific component types
            if isinstance(component, ModelInterface):
                self.model_interface = component
                # Connect resource monitor if available
                if self.resource_monitor:
                    await component.set_resource_monitor(self.resource_monitor)
                # Connect interactive session if available
                if self.interactive_session:
                    await component.set_interactive_session(self.interactive_session)
            
            # Connect CLI interface if appropriate
            elif isinstance(component, CommandLineInterface):
                self.cli_interface = component
                # Connect resource monitor
                if self.resource_monitor:
                    component.resource_monitor = self.resource_monitor
            
            logger.info(f"Component {component_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {str(e)}")
            return False
    
    async def connect_components(self) -> bool:
        """
        Connect registered components with each other.
        
        Returns:
            bool: True if all connections were successful, False otherwise
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            logger.info("Connecting components")
            
            # Connect model interface with CLI interface if both exist
            if self.model_interface and self.cli_interface:
                logger.info("Connecting model interface with CLI interface")
                # Pass chatbot to CLI
                if hasattr(self.model_interface, "chatbot"):
                    self.cli_interface.chatbot = self.model_interface.chatbot
                
            logger.info("Components connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect components: {str(e)}")
            return False
    
    async def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: The type of event to handle
            handler: The handler function or coroutine
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = set()
            
        self.event_handlers[event_type].add(handler)
        logger.debug(f"Registered handler for event type: {event_type}")
    
    async def trigger_event(self, event_type: str, data: Any) -> None:
        """
        Trigger an event to all registered handlers.
        
        Args:
            event_type: The type of event
            data: The event data
        """
        if event_type not in self.event_handlers:
            return
            
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for '{event_type}': {e}")
    
    async def get_component(self, component_id: str) -> Any:
        """
        Get a registered component by ID.
        
        Args:
            component_id: The component identifier
            
        Returns:
            The component instance if found, None otherwise
        """
        return self.components.get(component_id)
    
    async def get_component_by_type(self, component_type: Type) -> Any:
        """
        Get the first registered component of the given type.
        
        Args:
            component_type: The component type
            
        Returns:
            The component instance if found, None otherwise
        """
        for component in self.components.values():
            if isinstance(component, component_type):
                return component
        return None
    
    async def _periodic_status_check(self) -> None:
        """
        Periodically check the status of all components.
        """
        interval = self.config.get('status_check_interval', 60)
        try:
            while True:
                await asyncio.sleep(interval)
                await self._check_components_health()
        except asyncio.CancelledError:
            logger.debug("Status check task cancelled")
        except Exception as e:
            logger.error(f"Error in periodic status check: {str(e)}")
    
    async def _check_components_health(self) -> Dict[str, str]:
        """
        Check the health of all registered components.
        
        Returns:
            Dict[str, str]: Component health status
        """
        status = {}
        
        try:
            # Check resource status if monitor is available
            if self.resource_monitor:
                metrics = self.resource_monitor.get_metrics()
                if metrics.memory_usage > 90:
                    logger.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
                    # Try to optimize resources
                    if hasattr(self.resource_monitor, "optimize_resources"):
                        await self.resource_monitor.optimize_resources()
            
            # Check individual components
            for component_id, component in self.components.items():
                # Check if component has a health check method
                if hasattr(component, "health_check") and callable(getattr(component, "health_check")):
                    try:
                        health = await component.health_check() if asyncio.iscoroutinefunction(component.health_check) else component.health_check()
                        status[component_id] = health
                    except Exception as e:
                        logger.warning(f"Health check failed for component {component_id}: {str(e)}")
                        status[component_id] = "error"
                else:
                    # Simple presence check
                    status[component_id] = "present"
                    
        except Exception as e:
            logger.error(f"Error checking component health: {str(e)}")
            
        return status
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the integration and components.
        
        Returns:
            Dict[str, Any]: Status information
        """
        status = {
            "initialized": self.initialized,
            "interactive_mode": self.interactive_session is not None,
            "resource_monitoring": self.resource_monitor is not None,
            "components": {},
            "timestamp": time.time()
        }
        
        # Add component status
        for component_id, component in self.components.items():
            component_status = {"type": component.__class__.__name__}
            
            # Add detailed status if available
            if hasattr(component, "get_status") and callable(getattr(component, "get_status")):
                try:
                    detailed_status = await component.get_status() if asyncio.iscoroutinefunction(component.get_status) else component.get_status()
                    component_status.update(detailed_status)
                except Exception as e:
                    logger.warning(f"Failed to get status for component {component_id}: {str(e)}")
                    component_status["status"] = "error"
                    component_status["error"] = str(e)
            
            # Add active state if available
            if hasattr(component, "is_active"):
                component_status["active"] = component.is_active
                
            status["components"][component_id] = component_status
        
        # Add resource monitor status if available
        if self.resource_monitor:
            try:
                metrics = self.resource_monitor.get_metrics()
                status["resources"] = {
                    "memory_usage": metrics.memory_usage,
                    "cpu_usage": metrics.cpu_usage,
                    "disk_usage": metrics.disk_usage
                }
                if metrics.gpu_usage is not None:
                    status["resources"]["gpu_usage"] = metrics.gpu_usage
            except Exception as e:
                logger.warning(f"Failed to get resource metrics: {str(e)}")
        
        return status
    
    async def cleanup(self) -> None:
        """
        Clean up all components and resources.
        """
        logger.info("Cleaning up component integration")
        
        # Cancel status tasks
        for task in self.status_tasks:
            if not task.done():
                task.cancel()
        
        if self.status_tasks:
            try:
                await asyncio.gather(*self.status_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                pass
        
        # Clean up components in reverse registration order
        component_ids = list(self.components.keys())
        component_ids.reverse()
        
        for component_id in component_ids:
            component = self.components[component_id]
            try:
                logger.info(f"Cleaning up component: {component_id}")
                
                # Check for various cleanup methods
                if hasattr(component, "cleanup") and callable(getattr(component, "cleanup")):
                    if asyncio.iscoroutinefunction(component.cleanup):
                        await component.cleanup()
                    else:
                        component.cleanup()
                elif hasattr(component, "shutdown") and callable(getattr(component, "shutdown")):
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown()
                elif hasattr(component, "close") and callable(getattr(component, "close")):
                    if asyncio.iscoroutinefunction(component.close):
                        await component.close()
                    else:
                        component.close()
                        
            except Exception as e:
                logger.error(f"Error cleaning up component {component_id}: {str(e)}")
        
        # Clear components
        self.components.clear()
        
        # Clean up resource monitor
        if self.resource_monitor and hasattr(self.resource_monitor, "stop_monitoring"):
            try:
                await self.resource_monitor.stop_monitoring()
            except Exception as e:
                logger.error(f"Error stopping resource monitor: {str(e)}")
        
        # Clean up interactive session
        if self.interactive_session:
            try:
                await self.interactive_session.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down interactive session: {str(e)}")
        
        logger.info("Component integration cleanup completed")

class ComponentFactory:
    """
    Factory for creating and initializing system components.
    """
    
    @classmethod
    async def create_model_interface(cls, model_path: Optional[str] = None, 
                                  config: Dict[str, Any] = None) -> ModelInterface:
        """
        Create and initialize a model interface.
        
        Args:
            model_path: Path to the model file
            config: Configuration dictionary
            
        Returns:
            The initialized model interface
        """
        model_interface = ModelInterface(model_path=model_path, config=config)
        await model_interface.initialize()
        return model_interface
    
    @classmethod
    async def create_cli_interface(cls, chatbot_interface: Any, 
                               resource_monitor: Any = None) -> CommandLineInterface:
        """
        Create a CLI interface.
        
        Args:
            chatbot_interface: The chatbot interface to use
            resource_monitor: Optional resource monitor
            
        Returns:
            The initialized CLI interface
        """
        cli = CommandLineInterface(chatbot=chatbot_interface, resource_monitor=resource_monitor)
        return cli
    
    @classmethod
    async def load_user_interfaces(cls, integration: ComponentIntegration) -> Dict[str, Any]:
        """
        Load and register all user interfaces.
        
        Args:
            integration: The component integration instance
            
        Returns:
            Dict of registered UI components
        """
        ui_components = {}
        
        # Register CLI interface if available and there's a model interface
        if integration.model_interface and hasattr(integration.model_interface, "chatbot"):
            try:
                cli = await cls.create_cli_interface(
                    integration.model_interface.chatbot,
                    integration.resource_monitor
                )
                await integration.register_component("cli", cli)
                ui_components["cli"] = cli
            except Exception as e:
                logger.error(f"Failed to create CLI interface: {str(e)}")
        
        return ui_components
