import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import os

from core.constants import InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class UIMode(Enum):
    """Available UI modes for the system"""
    CLI = "cli"             # Command-line interface
    WEB = "web"             # Web-based interface
    API = "api"             # API-only interface
    DESKTOP = "desktop"     # Desktop application
    MOBILE = "mobile"       # Mobile interface


class UITheme(Enum):
    """Available UI themes"""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
    CUSTOM = "custom"


class UIManager:
    """Central manager for all user interfaces in the system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.active_interfaces: Dict[str, Any] = {}
        self.session = None
        self._interrupt_requested = False
        self.default_mode = UIMode(self.config.get("default_mode", "cli"))
        self.theme = UITheme(self.config.get("theme", "system"))
        self._status_subscribers = []
        
    async def start(self, mode: Optional[UIMode] = None) -> bool:
        """Start the UI system with the specified or default mode"""
        try:
            selected_mode = mode or self.default_mode
            logger.info(f"Starting UI in {selected_mode.value} mode")
            
            # Create interactive session for UI operations
            self.session = InteractiveSession(
                session_id="ui_manager",
                config=InteractiveConfig(
                    timeout=300,
                    persistent_state=True,
                    safe_mode=True
                )
            )
            
            await self.session.__aenter__()
            
            result = False
            if selected_mode == UIMode.CLI:
                result = await self.start_cli()
            elif selected_mode == UIMode.WEB:
                result = await self.start_web()
            elif selected_mode == UIMode.API:
                result = await self.start_api()
            elif selected_mode == UIMode.DESKTOP:
                result = await self.start_desktop()
            elif selected_mode == UIMode.MOBILE:
                result = await self.start_mobile()
                
            if not result:
                logger.error(f"Failed to start UI in {selected_mode.value} mode")
                
            return result
            
        except Exception as e:
            logger.error(f"Error starting UI: {e}")
            if self.session:
                await self.session.__aexit__(None, None, None)
            return False
            
    async def stop(self) -> None:
        """Stop all active UI interfaces"""
        logger.info("Stopping UI system")
        self._interrupt_requested = True
        
        stop_tasks = []
        for interface_name, interface in self.active_interfaces.items():
            if hasattr(interface, 'stop'):
                logger.info(f"Stopping {interface_name} interface")
                stop_tasks.append(interface.stop())
                
        # Wait for all interfaces to stop
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
        self.active_interfaces.clear()
        
        if self.session:
            await self.session.__aexit__(None, None, None)
            
    async def start_cli(self) -> bool:
        """Start command-line interface"""
        try:
            from ai_core.ui import CommandLineInterface
            from ai_core.chatbot.interface import ChatbotInterface
            
            # Get core components
            model, storage = await self._get_core_components()
            if model is None or storage is None:
                return False
                
            # Create interface
            interface = ChatbotInterface(
                model=model,
                storage=storage,
                interactive=True
            )
            
            # Create CLI
            cli = CommandLineInterface(interface)
            self.active_interfaces["cli"] = cli
            
            # Start the CLI in background
            cli_task = asyncio.create_task(cli.start())
            return True
            
        except ImportError as e:
            logger.error(f"Required CLI modules not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error starting CLI: {e}")
            return False
            
    async def start_web(self) -> bool:
        """Start web interface"""
        try:
            from web.server import WebServer
            
            # Get core components
            model, storage = await self._get_core_components()
            if model is None or storage is None:
                return False
            
            # Get web server configuration
            port = self.config.get("web_port", 8080)
            host = self.config.get("web_host", "0.0.0.0")
            
            # Create web server
            web_server = WebServer(
                model=model, 
                storage=storage,
                host=host,
                port=port
            )
            
            # Start web server
            await web_server.start()
            self.active_interfaces["web"] = web_server
            
            logger.info(f"Web interface started at http://{host}:{port}")
            return True
            
        except ImportError:
            logger.error("Web server modules not available")
            return False
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            return False
            
    async def start_api(self) -> bool:
        """Start API-only interface"""
        try:
            from api.server import APIServer
            
            # Get API configuration
            port = self.config.get("api_port", 8081)
            host = self.config.get("api_host", "0.0.0.0")
            
            # Create API server
            api_server = APIServer(
                host=host,
                port=port
            )
            
            # Start API server
            await api_server.start()
            self.active_interfaces["api"] = api_server
            
            logger.info(f"API interface started at http://{host}:{port}")
            return True
            
        except ImportError:
            logger.error("API server modules not available")
            return False
        except Exception as e:
            logger.error(f"Error starting API interface: {e}")
            return False
            
    async def start_desktop(self) -> bool:
        """Start desktop application interface"""
        logger.warning("Desktop UI not yet implemented")
        return False
        
    async def start_mobile(self) -> bool:
        """Start mobile application interface"""
        logger.warning("Mobile UI not yet implemented")
        return False
        
    async def _get_core_components(self):
        """Get core components needed for interfaces"""
        try:
            # Simple models for demonstration if full models aren't available
            model = None
            storage = None
            
            try:
                # Try to get main model
                from ai_core.models import SimpleNN
                model = SimpleNN(input_size=512, output_size=512, hidden_size=256)
            except ImportError:
                # Fallback model
                import torch.nn as nn
                class FallbackModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layer = nn.Linear(512, 512)
                        
                    def forward(self, x):
                        return self.layer(x)
                        
                    def generate_text(self, text, max_length=100):
                        return f"Response to: {text[:30]}..."
                        
                model = FallbackModel()
                
            try:
                # Try to get storage
                from ai_core.model_storage import ModelStorage
                storage = ModelStorage(storage_dir="./model_storage")
            except ImportError:
                # Fallback storage
                class FallbackStorage:
                    async def get_model_version(self):
                        return "0.1.0"
                        
                    async def store_feedback(self, feedback):
                        logger.debug(f"Storing feedback: {feedback}")
                        
                    async def persist_feedback(self, session_id, feedback):
                        logger.debug(f"Persisting feedback for session {session_id}")
                        
                storage = FallbackStorage()
                
            return model, storage
            
        except Exception as e:
            logger.error(f"Error getting core components: {e}")
            return None, None
            
    def subscribe_to_status(self, callback):
        """Subscribe to status updates"""
        self._status_subscribers.append(callback)
        
    def unsubscribe_from_status(self, callback):
        """Unsubscribe from status updates"""
        if callback in self._status_subscribers:
            self._status_subscribers.remove(callback)
            
    async def _update_status(self, status):
        """Update all subscribers with new status"""
        for subscriber in self._status_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(status)
                else:
                    subscriber(status)
            except Exception as e:
                logger.error(f"Error updating status subscriber: {e}")
