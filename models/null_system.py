"""
Null system that provides fallback functionality when the main system is unavailable.
"""
import logging

logger = logging.getLogger(__name__)

class NullSystem:
    """
    A fallback system that provides minimal functionality when the main system
    cannot be initialized.
    """
    
    def __init__(self):
        self._active = True
        self._agent_count = 0
        self._initialized = True
        self._active_learning = False
        logger.info("NullSystem initialized as fallback")
    
    def start(self):
        """Start the null system."""
        logger.info("NullSystem started (limited functionality)")
        return True
    
    def stop(self):
        """Stop the null system."""
        self._active = False
        logger.info("NullSystem stopped")
        return True
    
    def is_active(self):
        """Check if the system is active."""
        return self._active
    
    def get_agent_count(self):
        """Get the number of agents in the system."""
        return self._agent_count
    
    def get_status(self):
        """Get the system status."""
        return {
            "status": "limited",
            "mode": "fallback",
            "active": self._active,
            "agents": self._agent_count,
            "initialized": self._initialized,
            "learning_enabled": False
        }
    
    def process_command(self, command, *args, **kwargs):
        """Process a command."""
        logger.warning(f"Command '{command}' not available in fallback mode")
        return {
            "success": False,
            "message": "Command not available in fallback mode",
            "fallback": True
        }
