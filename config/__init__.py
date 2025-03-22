import logging
from dataclasses import dataclass

from .blockchain_config import BlockchainConfig
from .training_config import TrainingConfig
from .network_config import NetworkConfig

logger = logging.getLogger(__name__)

@dataclass
class Config:
    blockchain: BlockchainConfig
    training: TrainingConfig
    network: NetworkConfig
    
    def validate(self) -> None:
        """Enhanced configuration validation"""
        try:
            # Validate training config
            if self.training.batch_size <= 0:
                raise ValueError("batch_size must be positive")
                
            # Validate resource thresholds
            if hasattr(self.training, 'memory_limit'):
                memory = psutil.virtual_memory()
                if self.training.memory_limit > memory.total:
                    raise ValueError("Memory limit exceeds system capacity")
                    
            # Validate network settings
            if hasattr(self.network, 'port'):
                if self.network.port < 1024 or self.network.port > 65535:
                    raise ValueError("Port must be between 1024 and 65535")
                    
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def update_interactive(self) -> None:
        """Update all configurations interactively"""
        try:
            print("\nUpdating Configuration Interactively")
            
            print("\n=== Blockchain Configuration ===")
            self.blockchain.update_interactive()
            
            print("\n=== Training Configuration ===")
            self.training.update_interactive()
            
            print("\n=== Network Configuration ===")
            self.network.update_interactive()
            
            # Validate entire configuration
            self.validate()
            logger.info("All configurations updated successfully")
            
        except KeyboardInterrupt:
            logger.info("Configuration update cancelled by user")
            return
        except Exception as e:
            logger.error("Configuration update failed: %s", str(e))
            raise

    @classmethod
    def load(cls):
        return cls(
            blockchain=BlockchainConfig.from_env(),
            training=TrainingConfig.from_env(),
            network=NetworkConfig.from_env()
        )

    @classmethod
    def load_and_update(cls):
        """Enhanced config loading with validation"""
        try:
            config = cls.load()
            if input("\nUpdate configurations? (y/n): ").lower() == 'y':
                with InteractiveSession() as session:
                    config.update_interactive()
                    config.validate()
            return config
        except Exception as e:
            logger.error(f"Config loading failed: {e}")
            raise

__all__ = ['Config']
