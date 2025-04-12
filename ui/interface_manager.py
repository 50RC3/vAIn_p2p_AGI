"""
User Interface Manager for handling various UI types.
"""
import logging
from typing import Dict, List, Any, Optional
import threading
import time
import asyncio

logger = logging.getLogger(__name__)

class UserInterfaceManager:
    """
    Manages different user interfaces and their connections to the system.
    Provides a unified way to interact with various UI implementations.
    """
    
    def __init__(self):
        self.interfaces = {}
        self.active_interface = None
        self.system = None
        self._running = False
        self._event_thread = None
        self.component_integration = None
        
    def register_interface(self, interface):
        """Register a UI implementation with the manager."""
        if hasattr(interface, 'name'):
            name = interface.name
        else:
            name = interface.__class__.__name__
            
        self.interfaces[name] = interface
        
        if self.active_interface is None:
            self.active_interface = name
            
        logger.debug(f"Registered interface: {name}")
        
    def connect_system(self, system):
        """Connect the multi-agent system to the UI manager."""
        self.system = system
        
        # Connect the system to all interfaces
        for interface_name, interface in self.interfaces.items():
            if hasattr(interface, 'connect_system'):
                interface.connect_system(system)
            logger.debug(f"Connected system to interface: {interface_name}")
    
    def connect_integration(self, integration):
        """Connect component integration to UI manager."""
        self.component_integration = integration
        logger.info("Connected component integration to UI manager")
    
    async def start_async(self):
        """Start all registered interfaces asynchronously."""
        for name, interface in self.interfaces.items():
            try:
                if hasattr(interface, 'start'):
                    if asyncio.iscoroutinefunction(interface.start):
                        # Create task for async start methods
                        await interface.start()
                    else:
                        interface.start()
                    logger.info(f"Started interface: {name}")
            except Exception as e:
                logger.error(f"Failed to start interface {name}: {e}")
    
    def start(self):
        """Start all registered interfaces."""
        for name, interface in self.interfaces.items():
            try:
                if hasattr(interface, 'start'):
                    if asyncio.iscoroutinefunction(interface.start):
                        # Create task for async start methods
                        asyncio.create_task(interface.start())
                    else:
                        interface.start()
                    logger.info(f"Started interface: {name}")
            except Exception as e:
                logger.error(f"Failed to start interface {name}: {e}")
    
    def run_event_loop(self):
        """Run the event loop to handle UI events."""
        self._running = True
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()
        
        # If the active interface has its own event loop, use that
        active_ui = self.interfaces.get(self.active_interface)
        if active_ui and hasattr(active_ui, 'run_event_loop'):
            active_ui.run_event_loop()
        else:
            # Otherwise, just keep the main thread alive
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("User interrupted. Shutting down...")
                self._running = False
    
    def _event_loop(self):
        """Background event processing loop."""
        while self._running:
            # Process any background events
            for name, interface in self.interfaces.items():
                if hasattr(interface, 'process_events'):
                    interface.process_events()
            time.sleep(0.05)
    
    async def shutdown(self):
        """Shut down all interfaces."""
        self._running = False
        
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=2.0)
        
        # Shutdown interfaces asynchronously
        for name, interface in self.interfaces.items():
            try:
                if hasattr(interface, 'shutdown'):
                    if asyncio.iscoroutinefunction(interface.shutdown):
                        await interface.shutdown()
                    else:
                        interface.shutdown()
                    logger.info(f"Shutdown interface: {name}")
                elif hasattr(interface, 'stop'):
                    if asyncio.iscoroutinefunction(interface.stop):
                        await interface.stop()
                    else:
                        interface.stop()
                    logger.info(f"Stopped interface: {name}")
                elif hasattr(interface, 'close'):
                    if asyncio.iscoroutinefunction(interface.close):
                        await interface.close()
                    else:
                        interface.close()
                    logger.info(f"Closed interface: {name}")
            except Exception as e:
                logger.error(f"Error shutting down interface {name}: {e}")
                
        # Disconnect from component integration
        self.component_integration = None
        self.system = None
