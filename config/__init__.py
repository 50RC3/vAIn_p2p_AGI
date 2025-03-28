import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from core.interactive_session import InteractiveSession
except ImportError:
    InteractiveSession = None

from .blockchain_config import BlockchainConfig
from .training_config import TrainingConfig 
from .network_config import NetworkConfig

logger = logging.getLogger(__name__)

@dataclass
class Config:
    blockchain: Optional[BlockchainConfig] = None
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        hidden_size=256,
        num_layers=2,
        num_rounds=5,
        min_clients=2,
        clients_per_round=2
    ))
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig(
        node_env='development',
        port=8000,
        database_url='sqlite:///db.sqlite'
    ))

    def validate(self) -> None:
        """Enhanced configuration validation"""
        if self.blockchain:
            self.blockchain.validate_config()
        if hasattr(self.training, 'validate'):
            self.training.validate()
        self.network.validate()

    def update_interactive(self) -> None:
        """Update all configurations interactively"""
        if not InteractiveSession:
            logger.warning("Interactive session not available")
            return
            
        try:
            print("\nUpdating Configuration Interactively")
            
            print("\n=== Blockchain Configuration ===")
            if self.blockchain:
                self.blockchain.update_interactive()
            
            print("\n=== Training Configuration ===")
            self.training.update_interactive()
            
            print("\n=== Network Configuration ===")
            self.network.update_interactive()
            
            self.validate()
            logger.info("All configurations updated successfully")
            
        except KeyboardInterrupt:
            logger.info("Configuration update cancelled by user")
            return
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise

    @classmethod
    def load(cls) -> 'Config':
        try:
            return cls(
                blockchain=BlockchainConfig.from_env(),
                training=TrainingConfig.from_env(),
                network=NetworkConfig.from_env()
            )
        except ImportError:
            logger.warning("Some dependencies missing - install required packages first")
            return cls()

    @classmethod
    def load_and_update(cls) -> 'Config':
        """Enhanced config loading with validation"""
        try:
            config = cls.load()
            if input("\nUpdate configurations? (y/n): ").lower() == 'y':
                config.update_interactive()
                config.validate()
            return config
        except Exception as e:
            logger.error("Config loading failed: %s", str(e))
            raise

__all__ = ['Config']
