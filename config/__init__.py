"""
Configuration package for vAIn_p2p_AGI

This package handles all configuration management for the system,
including loading, validation, and interactive updating of configuration.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from core.constants import InteractionLevel
except ImportError:
    from enum import Enum
    class InteractionLevel(Enum):
        NONE = "none"
        MINIMAL = "minimal"
        NORMAL = "normal"
        VERBOSE = "verbose"

from .blockchain_config import BlockchainConfig
from .training_config import TrainingConfig 
from .network_config import NetworkConfig

# Import the system_config if available
try:
    from .system_config import SystemConfig, get_system_config
except ImportError:
    # Use minimal fallback if not available
    from dataclasses import dataclass, field
    
    @dataclass
    class SystemConfig:
        node_id: str = ""
        interactive: bool = True
        log_level: str = "INFO"
        
    def get_system_config():
        return SystemConfig()

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Central configuration container that holds all config components.
    """
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
    network: NetworkConfig = field(default_factory=NetworkConfig)
    system: SystemConfig = field(default_factory=get_system_config)
    
    # Add node identification attributes
    node_id: str = field(default_factory=lambda: os.getenv('NODE_ID', 'default_node_id'))
    interactive: bool = True

    def validate(self) -> bool:
        """Enhanced configuration validation"""
        try:
            # Use config_validator if available for comprehensive validation
            try:
                from tools.config_validator import ConfigValidator, ValidationResult
                
                # Validate system config
                system_result = ConfigValidator.validate("system", self.system.to_dict() 
                                                       if hasattr(self.system, 'to_dict') 
                                                       else {"node_id": self.system.node_id})
                if not system_result.valid:
                    logger.error(f"System configuration validation failed: {system_result.get_report()}")
                    return False
                    
                # Validate blockchain config if present
                if self.blockchain:
                    blockchain_dict = self.blockchain.__dict__ if hasattr(self.blockchain, '__dict__') else {}
                    blockchain_result = ConfigValidator.validate("blockchain", blockchain_dict)
                    if not blockchain_result.valid:
                        logger.error(f"Blockchain configuration validation failed: {blockchain_result.get_report()}")
                        return False
                
                # Validate training config
                training_dict = self.training.__dict__ if hasattr(self.training, '__dict__') else {}
                training_result = ConfigValidator.validate("training", training_dict)
                if not training_result.valid:
                    logger.error(f"Training configuration validation failed: {training_result.get_report()}")
                    return False
                
                # Validate network config
                network_dict = self.network.__dict__ if hasattr(self.network, '__dict__') else {}
                network_result = ConfigValidator.validate("network", network_dict)
                if not network_result.valid:
                    logger.error(f"Network configuration validation failed: {network_result.get_report()}")
                    return False
                    
                logger.info("All configurations validated successfully")
                return True
            
            except ImportError:
                # Fallback to basic validation
                if hasattr(self.system, 'validate') and not self.system.validate():
                    logger.error("System configuration validation failed")
                    return False
                    
                # Validate blockchain config if present
                if self.blockchain and hasattr(self.blockchain, 'validate_config'):
                    if not self.blockchain.validate_config():
                        logger.error("Blockchain configuration validation failed")
                        return False
                    
                # Validate training config
                if hasattr(self.training, 'validate'):
                    if not self.training.validate():
                        logger.error("Training configuration validation failed")
                        return False
                    
                # Validate network config
                if not self.network.validate():
                    logger.error("Network configuration validation failed")
                    return False
                    
                logger.info("All configurations validated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False

    def update_interactive(self) -> bool:
        """Update all configurations interactively"""
        if not self.interactive:
            logger.warning("Interactive mode disabled")
            return False
            
        try:
            print("\nUpdating Configuration Interactively")
            
            # Try to use the config_manager if available
            try:
                from tools.config_manager import ConfigManager
                config_manager = ConfigManager()
                return config_manager.unified_config_update()
            except ImportError:
                # Fallback to manual updates
                print("\n=== System Configuration ===")
                if hasattr(self.system, 'update_interactive'):
                    self.system.update_interactive()
                
                print("\n=== Blockchain Configuration ===")
                if self.blockchain and hasattr(self.blockchain, 'update_interactive'):
                    self.blockchain.update_interactive()
                
                print("\n=== Training Configuration ===")
                if hasattr(self.training, 'update_interactive'):
                    self.training.update_interactive()
                
                print("\n=== Network Configuration ===")
                if hasattr(self.network, 'update_interactive'):
                    self.network.update_interactive()
                
                valid = self.validate()
                if valid:
                    logger.info("All configurations updated successfully")
                return valid
                
        except KeyboardInterrupt:
            logger.info("Configuration update cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False

    @classmethod
    def load(cls, config_dir: Optional[str] = None):
        """Load configuration from path"""
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "config"
        
        blockchain = None
        try:
            blockchain = BlockchainConfig.from_env()
        except Exception as e:
            logger.warning(f"Failed to load blockchain config: {e}")
        
        try:
            training = TrainingConfig.from_env()
        except Exception as e:
            logger.warning(f"Failed to load training config: {e}")
            training = TrainingConfig()
            
        try:
            network = NetworkConfig.from_env()
        except Exception as e:
            logger.warning(f"Failed to load network config: {e}")
            network = NetworkConfig()
        
        # Load system config if available
        try:
            system_config_path = config_dir / "system.json"
            system = get_system_config(system_config_path)
        except Exception as e:
            logger.warning(f"Failed to load system config: {e}")
            system = get_system_config()
            
        return cls(
            blockchain=blockchain,
            training=training,
            network=network,
            system=system,
        )

    @classmethod
    def load_and_update(cls, interactive: bool = True):
        """Enhanced config loading with validation and optional interactive updates"""
        try:
            config = cls.load()
            config.interactive = interactive
            
            if interactive:
                if input("\nUpdate configurations? (y/n): ").lower() == 'y':
                    config.update_interactive()
            
            config.validate()
            return config
        except Exception as e:
            logger.error(f"Config loading failed: {e}")
            # Create minimal config as fallback
            return cls(interactive=interactive)
    
    def save(self, config_dir: Optional[str] = None) -> bool:
        """Save all configurations to disk"""
        try:
            config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "config"
            config_dir.mkdir(exist_ok=True, parents=True)
            
            # Save system config if it has a save method
            if hasattr(self.system, 'save'):
                self.system.save(config_dir / "system.json")
            
            # For other configs, manually save them
            if self.blockchain:
                blockchain_dict = self.blockchain.__dict__ if hasattr(self.blockchain, '__dict__') else {}
                with open(config_dir / "blockchain.json", "w") as f:
                    json.dump(blockchain_dict, f, indent=2)
            
            training_dict = self.training.__dict__ if hasattr(self.training, '__dict__') else {}
            with open(config_dir / "training.json", "w") as f:
                json.dump(training_dict, f, indent=2)
            
            network_dict = self.network.__dict__ if hasattr(self.network, '__dict__') else {}
            with open(config_dir / "network.json", "w") as f:
                json.dump(network_dict, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            return False

def get_config(interactive: bool = True) -> Config:
    """Get configuration instance with optional interactive setup
    
    Args:
        interactive: Whether to allow interactive configuration
        
    Returns:
        Config: Configured instance
    """
    try:
        config = Config.load()
        config.interactive = interactive
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.warning(f"Error loading full configuration: {e}")
        # Return minimal config
        minimal_config = Config()
        minimal_config.interactive = interactive
        return minimal_config

__all__ = ['Config', 'get_config', 'SystemConfig', 'get_system_config']
