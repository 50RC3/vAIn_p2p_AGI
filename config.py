import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

try:
    from core.interactive_utils import InteractiveConfig, InteractionLevel
    from core.constants import INTERACTION_TIMEOUTS
except ImportError:
    # Define minimal versions if not available
    class InteractionLevel:
        NORMAL = "normal"
    
    @dataclass
    class InteractiveConfig:
        timeout: int = 300
        persistent_state: bool = True
        safe_mode: bool = True
    
    INTERACTION_TIMEOUTS = {
        "default": 300,
        "confirmation": 60,
        "config": 300,
    }

logger = logging.getLogger(__name__)

@dataclass
class Config:
    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        self._config_path = Path(__file__).parent / 'config.json'
        self._session = None
        self.last_validation = None
        self._interrupt_requested = False
        
        # Load env vars
        try:
            load_dotenv()
        except:
            pass
        
        # Node identification
        self.node_id: str = os.getenv('NODE_ID', 'default_node_id')
        
        # Network configuration
        self.network: Dict[str, Any] = {
            'udp': {
                'port': int(os.getenv('UDP_PORT', '8468'))
            },
            'dht': {
                'bootstrap_nodes': os.getenv('DHT_BOOTSTRAP_NODES', '').split(',') 
            },
            'secret_key': os.getenv('NETWORK_SECRET', 'default_secret')
        }
        
        # Training parameters
        self.batch_size: int = int(os.getenv('BATCH_SIZE', 32))
        self.learning_rate: float = float(os.getenv('LEARNING_RATE', 0.001))
        self.num_epochs: int = int(os.getenv('NUM_EPOCHS', 100))
        
        # Model parameters
        self.hidden_size: int = int(os.getenv('HIDDEN_SIZE', 256))
        self.num_layers: int = int(os.getenv('NUM_LAYERS', 4))
        
        # Federated learning parameters
        self.num_rounds: int = int(os.getenv('NUM_ROUNDS', 50))
        self.min_clients: int = int(os.getenv('MIN_CLIENTS', 3))
        self.clients_per_round: int = int(os.getenv('CLIENTS_PER_ROUND', 10))
        
        # Chatbot parameters
        self.chatbot: Dict[str, Any] = {
            'max_context': int(os.getenv('CHATBOT_MAX_CONTEXT', 1024)),
            'response_temp': float(os.getenv('CHATBOT_RESPONSE_TEMP', 0.7)),
            'top_p': float(os.getenv('CHATBOT_TOP_P', 0.9)),
            'max_tokens': int(os.getenv('CHATBOT_MAX_TOKENS', 256))
        }
        
        # RL parameters
        self.rl: Dict[str, Any] = {
            'gamma': float(os.getenv('RL_GAMMA', 0.99)),
            'learning_rate': float(os.getenv('RL_LEARNING_RATE', 0.001)),
            'batch_size': int(os.getenv('RL_BATCH_SIZE', 32)),
            'update_interval': int(os.getenv('RL_UPDATE_INTERVAL', 100)),
            'memory_size': int(os.getenv('RL_MEMORY_SIZE', 10000))
        }

    async def update_interactive(self) -> bool:
        """Interactive configuration update with validation and recovery"""
        try:
            # For compatibility with both modules
            from config import get_config
            config_module = get_config(interactive=self.interactive)
            config_module.update_interactive()
            return True
        except ImportError:
            # Fall back to local implementation
            logger.info("Using standalone config update")
            self._display_current_config()
            return await self._update_config_interactive()
    
    async def _update_config_interactive(self) -> bool:
        """Local implementation of interactive config update"""
        print("\nUpdate configuration? (y/n): ")
        response = input()
        if response.lower() != 'y':
            return False
            
        # Update basic parameters
        self.batch_size = int(input(f"Batch size [{self.batch_size}]: ") or self.batch_size)
        self.learning_rate = float(input(f"Learning rate [{self.learning_rate}]: ") or self.learning_rate)
        self.num_epochs = int(input(f"Epochs [{self.num_epochs}]: ") or self.num_epochs)
        
        return True
            
    def validate_config(self) -> bool:
        """Enhanced configuration validation with bounds checking"""
        # Add some basic validation
        if self.batch_size <= 0:
            logger.error("Batch size must be positive")
            return False
            
        if not (0 < self.learning_rate < 1):
            logger.error("Learning rate must be between 0 and 1")
            return False
            
        return True

    def _display_current_config(self):
        """Display current configuration values"""
        print("\nNetwork Configuration:")
        print(f"UDP Port: {self.network['udp']['port']}")
        print(f"DHT Bootstrap Nodes: {len(self.network['dht']['bootstrap_nodes'])}")
        
        print("\nTraining Configuration:")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        
        print("\nModel Configuration:")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Number of Layers: {self.num_layers}")
        
        print("\nChatbot Configuration:")
        for k, v in self.chatbot.items():
            print(f"{k}: {v}")
            
        print("\nRL Configuration:")
        for k, v in self.rl.items():
            print(f"{k}: {v}")

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True

# Maintain backwards compatibility for non-package imports
def get_config(interactive: bool = True) -> Config:
    """Get configuration object compatible with both module and standalone versions"""
    try:
        # Try importing from the package first
        from config import get_config as pkg_get_config
        return pkg_get_config(interactive)
    except ImportError:
        # Fall back to local implementation
        return Config(interactive=interactive)

# Chatbot Configuration
CHATBOT_CONFIG = {
    'max_context_length': 1024,
    'response_temp': 0.7,
    'top_p': 0.9,
    'max_tokens': 256
}

# Reinforcement Learning
RL_CONFIG = {
    'gamma': 0.99,
    'learning_rate': 0.001,
    'batch_size': 32,
    'update_interval': 100,
    'memory_size': 10000,
    'min_samples': 1000
}
