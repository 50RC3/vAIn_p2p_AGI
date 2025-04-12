"""
Integration Helper for vAIn P2P AGI system.
Provides simplified methods to connect system components.
"""
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Set, Callable, Union, Type

from core.component_integration import ComponentIntegration, ComponentFactory
from core.interactive_config import InteractionLevel, InteractiveConfig
from training.model_interface import ModelInterface
from ai_core.ui.cli_interface import CommandLineInterface
from ui.interface_manager import UserInterfaceManager

logger = logging.getLogger(__name__)

class IntegrationHelper:
    """
    Provides simplified methods to integrate different parts of the system.
    Wraps the ComponentIntegration class with higher-level functions.
    """
    
    @staticmethod
    async def setup_complete_system(config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Sets up the complete vAIn system with all necessary components.
        
        Args:
            config: System configuration
            
        Returns:
            Dict containing the integration and all components
        """
        # Default configuration
        if config is None:
            config = {
                'interactive_mode': True,
                'resource_monitoring': True,
                'status_check_interval': 60,
                'model': {
                    'path': os.path.join('models', 'default_model.pt'),
                    'batch_size': 16,
                    'learning_rate': 0.001
                }
            }
        
        # Create component integration
        integration = ComponentIntegration(config)
        await integration.initialize()
        
        # Create and register components
        model_interface = await ComponentFactory.create_model_interface(
            model_path=config['model'].get('path'),
            config=config['model']
        )
        await integration.register_component("model", model_interface)
        
        # Load user interfaces
        ui_components = await ComponentFactory.load_user_interfaces(integration)
        
        # Connect components
        await integration.connect_components()
        
        # Setup UI manager if needed
        ui_manager = None
        if config.get('use_ui_manager', True):
            ui_manager = UserInterfaceManager()
            
            # Connect to component integration
            ui_manager.connect_integration(integration)
            
            # Register CLI interface if available
            if "cli" in ui_components:
                ui_manager.register_interface(ui_components["cli"])
                
            # Connect to system (in this case our model interface)
            ui_manager.connect_system(model_interface)
        
        # Return all components
        result = {
            "integration": integration,
            "model_interface": model_interface,
            "ui_components": ui_components,
            "ui_manager": ui_manager
        }
        
        logger.info("System setup completed successfully")
        return result
    
    @staticmethod
    async def connect_resource_monitor(integration: ComponentIntegration, components: Dict[str, Any]) -> bool:
        """
        Connects the resource monitor to all relevant components.
        
        Args:
            integration: The component integration instance
            components: Dictionary of components to connect
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not integration.resource_monitor:
            logger.warning("No resource monitor available to connect")
            return False
            
        try:
            for name, component in components.items():
                # Check if component can have a resource monitor
                if hasattr(component, 'set_resource_monitor'):
                    if asyncio.iscoroutinefunction(component.set_resource_monitor):
                        await component.set_resource_monitor(integration.resource_monitor)
                    else:
                        component.resource_monitor = integration.resource_monitor
                    logger.debug(f"Connected resource monitor to {name}")
                    
            return True
        except Exception as e:
            logger.error(f"Failed to connect resource monitor: {str(e)}")
            return False
    
    @staticmethod
    async def connect_interactive_sessions(integration: ComponentIntegration, components: Dict[str, Any]) -> bool:
        """
        Connects interactive sessions to all relevant components.
        
        Args:
            integration: The component integration instance
            components: Dictionary of components to connect
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not integration.interactive_session:
            logger.warning("No interactive session available to connect")
            return False
            
        try:
            for name, component in components.items():
                # Check if component can have an interactive session
                if hasattr(component, 'set_interactive_session'):
                    if asyncio.iscoroutinefunction(component.set_interactive_session):
                        await component.set_interactive_session(integration.interactive_session)
                    else:
                        component.interactive_session = integration.interactive_session
                    logger.debug(f"Connected interactive session to {name}")
                    
            return True
        except Exception as e:
            logger.error(f"Failed to connect interactive sessions: {str(e)}")
            return False
    
    @staticmethod
    async def setup_event_handlers(integration: ComponentIntegration, components: Dict[str, Any]) -> bool:
        """
        Set up event handlers between components.
        
        Args:
            integration: The component integration instance
            components: Dictionary of components to connect
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set up event handlers for model interface
            model_interface = components.get("model_interface")
            if model_interface:
                # Register system-wide handlers
                await integration.register_event_handler("model_training_started", 
                                                     lambda data: logger.info(f"Training started: {data}"))
                await integration.register_event_handler("model_training_completed", 
                                                     lambda data: logger.info(f"Training completed: {data}"))
                
                # Connect model events to UI
                ui_manager = components.get("ui_manager")
                if ui_manager and hasattr(model_interface, "register_handler"):
                    # Example: Connect model training events to UI
                    for event in ["training_started", "training_completed", "prediction_completed", "error"]:
                        model_interface.register_handler(event, 
                            lambda data, event_name=event: asyncio.create_task(
                                integration.trigger_event(f"model_{event_name}", data)
                            )
                        )
                    logger.info("Set up model event handlers")
                    
            # Connect chatbot interfaces to UI
            cli_interface = None
            for name, component in components.get("ui_components", {}).items():
                if name == "cli" and hasattr(component, "chatbot") and component.chatbot:
                    cli_interface = component
                    break
                    
            if cli_interface and cli_interface.chatbot:
                # Connect chatbot events to integration
                if hasattr(cli_interface.chatbot, "register_handler"):
                    for event in ["message_processed", "feedback_stored", "session_started", "session_cleared"]:
                        cli_interface.chatbot.register_handler(event, 
                            lambda data, event_name=event: asyncio.create_task(
                                integration.trigger_event(f"chatbot_{event_name}", data)
                            )
                        )
                    logger.info("Set up chatbot event handlers")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up event handlers: {str(e)}")
            return False
    
    @staticmethod
    async def start_ui_components(ui_components: Dict[str, Any]) -> None:
        """
        Start all UI components.
        
        Args:
            ui_components: Dictionary of UI components to start
        """
        for name, component in ui_components.items():
            if hasattr(component, 'start'):
                if asyncio.iscoroutinefunction(component.start):
                    # Start async UI components
                    asyncio.create_task(component.start())
                else:
                    # Start synchronous UI components
                    component.start()
                logger.info(f"Started UI component: {name}")
    
    @staticmethod
    async def cleanup_system(integration: ComponentIntegration, components: Dict[str, Any]) -> None:
        """
        Cleanup all components and integration.
        
        Args:
            integration: The component integration instance
            components: Additional components to clean up
        """
        # First cleanup components not managed by integration
        for name, component in components.items():
            if not component:  # Skip None components
                continue
                
            if hasattr(component, 'cleanup') and asyncio.iscoroutinefunction(component.cleanup):
                try:
                    await component.cleanup()
                    logger.debug(f"Cleaned up component: {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {str(e)}")
            elif hasattr(component, 'shutdown') and asyncio.iscoroutinefunction(component.shutdown):
                try:
                    await component.shutdown() 
                    logger.debug(f"Shut down component: {name}")
                except Exception as e:
                    logger.warning(f"Error shutting down {name}: {str(e)}")
        
        # Then cleanup integration (which will cleanup its registered components)
        if integration:
            await integration.cleanup()
            
    @staticmethod 
    async def register_universal_components(integration: ComponentIntegration, components: Dict[str, Any]) -> None:
        """
        Register all components with the integration system.
        
        Args:
            integration: The component integration instance  
            components: Dictionary of components to register
        """
        # Register components that aren't already managed by integration
        for name, component in components.items():
            if name != "integration" and name != "ui_components" and component is not None:
                # Skip components that might already be registered
                if name not in integration.components and name != "model":
                    await integration.register_component(name, component)
                    logger.debug(f"Registered component: {name}")
