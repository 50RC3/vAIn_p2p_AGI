"""
Base class for all user interfaces.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseUI(ABC):
    """Abstract base class for all UI implementations."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.system = None
    
    def connect_system(self, system):
        """Connect the multi-agent system to this UI."""
        self.system = system
    
    @abstractmethod
    def start(self):
        """Initialize and start the UI."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean up and shut down the UI."""
        pass
    
    def process_events(self):
        """Process any pending events in the UI."""
        pass
    
    def display_message(self, message, level="info"):
        """Display a message in the UI."""
        log_methods = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
        }
        
        log_method = log_methods.get(level.lower(), logger.info)
        log_method(message)
