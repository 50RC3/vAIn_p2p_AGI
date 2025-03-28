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

from core.interactive_utils import InteractiveConfig, InteractionLevel 
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class Config:
    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        self._config_path = Path(__file__).parent / 'config.json'
        self._session: Optional[InteractiveSession] = None
        self.last_validation: Optional[bool] = None
        self._interrupt_requested = False
        
        # Load env vars
        load_dotenv()
        
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
        self.num_epochs: int = 100
        
        # Model parameters
        self.hidden_size: int = 256
        self.num_layers: int = 4
        
        # Federated learning parameters
        self.num_rounds: int = 50
        self.min_clients: int = 3
        self.clients_per_round: int = 10
        
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
            self._session = InteractiveSession(
                level=InteractionLevel.NORMAL if self.interactive else InteractionLevel.NONE,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["config"],
                    persistent_state=True,
                    safe_mode=True,
                    recovery_enabled=True,
                    max_cleanup_wait=30
                )
            )

            async with self._session:
                # Try to restore previous state
                saved_state = await self._load_saved_state()
                if saved_state:
                    if await self._confirm_restore(saved_state):
                        self._restore_state(saved_state)
                        logger.info("Configuration restored from saved state")

                print("\nCurrent Configuration:")
                self._display_current_config()

                if not await self._session.confirm_with_timeout(
                    "\nUpdate configuration? (y/n): ",
                    timeout=INTERACTION_TIMEOUTS["confirmation"]
                ):
                    return False

                # Update sections interactively
                await self._update_network_config()
                await self._update_training_config()
                await self._update_model_config()
                await self._update_chatbot_config()
                await self._update_rl_config()

                # Validate and save
                if self.validate_config():
                    await self._save_state()
                    logger.info("Configuration updated and saved successfully")
                    return True
                return False

        except asyncio.TimeoutError:
            logger.error("Configuration update timed out")
            return False
        except Exception as e:
            logger.error(f"Configuration update failed: {str(e)}")
            return False
        finally:
            await self._cleanup()

    def validate_config(self) -> bool:
        """Enhanced configuration validation with bounds checking"""
        try:
            # Network validation
            assert 1024 <= self.network['udp']['port'] <= 65535, "Invalid UDP port"
            assert self.network['secret_key'], "Missing network secret key"

            # Training validation  
            assert 1 <= self.batch_size <= 1024, "Invalid batch size"
            assert 0 < self.learning_rate <= 1, "Invalid learning rate"
            assert 1 <= self.num_epochs <= 1000, "Invalid epoch count"
            
            # Model validation
            assert 16 <= self.hidden_size <= 2048, "Invalid hidden size"
            assert 1 <= self.num_layers <= 32, "Invalid layer count"

            # Chatbot validation
            assert self.chatbot['max_context'] > 0, "Invalid max context length"
            assert 0 < self.chatbot['response_temp'] <= 1, "Invalid response temperature" 
            assert 0 < self.chatbot['top_p'] <= 1, "Invalid top_p value"
            assert 0 < self.chatbot['max_tokens'] <= 2048, "Invalid max tokens"

            # RL validation
            assert 0 < self.rl['gamma'] <= 1, "Invalid gamma value"
            assert 0 < self.rl['learning_rate'] <= 1, "Invalid learning rate"
            assert 0 < self.rl['batch_size'] <= 1024, "Invalid batch size"
            assert self.rl['memory_size'] >= self.rl['batch_size'], "Memory size too small"

            self.last_validation = True
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            self.last_validation = False
            return False

    async def _load_saved_state(self) -> Optional[Dict[str, Any]]:
        """Load saved configuration state"""
        try:
            if self._config_path.exists():
                async with self._session.file_operation():
                    with open(self._config_path) as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load saved state: {str(e)}")
        return None

    async def _save_state(self) -> bool:
        """Save current configuration state"""
        try:
            async with self._session.file_operation():
                with open(self._config_path, 'w') as f:
                    json.dump(self._get_serializable_state(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            return False

    async def _cleanup(self):
        """Cleanup resources"""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None

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
