import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, TextIO, Union
from pathlib import Path
from dataclasses import dataclass
try:
    from dotenv import load_dotenv
except ImportError:
    from typing import Any, Optional, TextIO, Union # Ensure necessary types are imported
    from pathlib import Path # Ensure Path is imported

    def load_dotenv(
        dotenv_path: Union[str, Path, None] = None,
        stream: Optional[TextIO] = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: Optional[str] = "utf-8",
        **kwargs: Any,
    ) -> bool:
        """
        Stub implementation of load_dotenv if dotenv is not available.
        Matches the signature of the real function but does nothing.

        Returns:
            bool: Always returns False.
        """
        # Mark arguments as used to satisfy linters/type checkers while maintaining signature
        _ = dotenv_path
        _ = stream
        _ = verbose
        _ = override
        _ = interpolate
        _ = encoding
        _ = kwargs
        # This stub does nothing and indicates that .env loading did not happen.
        return False

try:
    from core.interactive_utils import InteractiveConfig, InteractionLevel
    from core.constants import INTERACTION_TIMEOUTS
except ImportError:
    # Define minimal versions if not available
    class InteractionLevel:
        NORMAL = "normal"

    @dataclass
    class _FallbackInteractiveConfig:  # Renamed to avoid collision
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
        except ImportError:
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
            'max_context': int(os.getenv('CHATBOT_MAX_CONTEXT', '1024')),
            'response_temp': float(os.getenv('CHATBOT_RESPONSE_TEMP', '0.7')),
            'top_p': float(os.getenv('CHATBOT_TOP_P', '0.9')),
            'max_tokens': int(os.getenv('CHATBOT_MAX_TOKENS', '256'))
        }
        
        # RL parameters
        self.rl: Dict[str, Any] = {
            'gamma': float(os.getenv('RL_GAMMA', '0.99')),
            'learning_rate': float(os.getenv('RL_LEARNING_RATE', '0.001')),
            'batch_size': int(os.getenv('RL_BATCH_SIZE', '32')),
            'update_interval': int(os.getenv('RL_UPDATE_INTERVAL', '100')),
            'memory_size': int(os.getenv('RL_MEMORY_SIZE', '10000'))
        }

    async def update_interactive(self) -> bool:
        """Interactive configuration update with validation and recovery"""
        try:
            # Use the function directly from this module
            config_module = get_config(interactive=self.interactive)
            # Avoid recursive call by checking if it's the same instance
            if config_module is not self:
                await config_module.update_interactive()
            return True
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug("Error using module config: %s", e)
            # Fall back to local implementation
            logger.info("Using standalone config update")
            self._display_current_config()
            return await self._update_config_interactive()
    
    async def _update_config_interactive(self) -> bool:
        """Local implementation of interactive config update"""
        try:
            print("\nUpdate configuration? (y/n): ")
            response = input()
            if response.lower() != 'y':
                return False
                
            # Update basic parameters
            self.batch_size = int(input(f"Batch size [{self.batch_size}]: ") or self.batch_size)
            self.learning_rate = float(input(f"Learning rate [{self.learning_rate}]: ") or self.learning_rate)
            self.num_epochs = int(input(f"Epochs [{self.num_epochs}]: ") or self.num_epochs)
            
            # Validate after updates
            if not self.validate_config():
                logger.error("Configuration validation failed")
                return False
                
            logger.info("Configuration updated successfully")
            return True
        except ValueError as e:
            logger.error("Invalid input: %s", e)
            return False
        except KeyboardInterrupt:
            logger.error("Configuration update canceled by user")
            return False
        except EOFError:
            logger.error("Input stream ended unexpectedly")
            return False
        except IOError as e:
            logger.error("I/O error during configuration update: %s", e)
            return False
            
    def validate_config(self) -> bool:
        """Enhanced configuration validation with bounds checking"""
        # Add some basic validation
        if self.batch_size <= 0:
            logger.error("Batch size must be positive")
            return False
            
        if not (0 < self.learning_rate < 1):
            logger.error("Learning rate must be between 0 and 1")
            return False
            
        # Add validation for RL parameters
        if self.rl['gamma'] <= 0 or self.rl['gamma'] >= 1:
            logger.error("RL gamma must be between 0 and 1 (exclusive)")
            return False
            
        return True

    def _display_current_config(self) -> None:
        """Display current configuration values"""
        def display_section(title: str, items: Dict[str, Any]) -> None:
            print(f"\n{title}:")
            for k, v in items.items():
                print(f"{k}: {v}")

        try:
            display_section("Network Configuration", {
                "UDP Port": self.network.get('udp', {}).get('port', 'N/A'),
                "DHT Bootstrap Nodes": len(self.network.get('dht', {}).get('bootstrap_nodes', []))
            })
        except AttributeError as e:
            logger.error("Error displaying network config: %s", e)

        try:
            display_section("Training Configuration", {
                "Batch Size": self.batch_size,
                "Learning Rate": self.learning_rate,
                "Epochs": self.num_epochs
            })
        except AttributeError as e:
            logger.error("Error displaying training config: %s", e)

        try:
            display_section("Model Configuration", {
                "Hidden Size": self.hidden_size,
                "Number of Layers": self.num_layers
            })
        except AttributeError as e:
            logger.error("Error displaying model config: %s", e)

        try:
            display_section("Chatbot Configuration", self.chatbot)
        except AttributeError as e:
            logger.error("Error displaying chatbot config: %s", e)

        try:
            display_section("RL Configuration", self.rl)
        except AttributeError as e:
            logger.error("Error displaying RL config: %s", e)

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        self._interrupt_requested = True

# Maintain backwards compatibility for non-package imports
def get_config(interactive: bool = True) -> Config:
    """Get configuration object compatible with both module and standalone versions"""
    try:
        # Check if we're being imported as a package
        import sys
        if 'config' in sys.modules and sys.modules['config'] is not sys.modules[__name__]:
            # Get config from the package
            return sys.modules['config'].get_config(interactive)
        # Fall back to local implementation
        return Config(interactive=interactive)
    except (ImportError, AttributeError):
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
