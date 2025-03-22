from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
import json
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    hidden_size: int
    memory_size: int 
    memory_vector_dim: int
    num_heads: int
    num_layers: int
    input_size: int
    num_agents: int
    learning_rate: float = 0.001

    # Add new fields with bounds validation
    max_memory_size: int = 1000000
    batch_size: int = 32
    dropout: float = 0.1
    checkpoint_interval: int = 100
    backup_path: str = field(default="backups")
    config_bounds: Dict = field(default_factory=lambda: {
        'hidden_size': (64, 2048),
        'memory_size': (1000, 1000000),
        'memory_vector_dim': (16, 512),
        'num_heads': (1, 32),
        'num_layers': (1, 24),
        'batch_size': (1, 512),
        'learning_rate': (0.0001, 0.1),
        'dropout': (0.0, 0.5)
    })

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate_config()
        os.makedirs(self.backup_path, exist_ok=True)

    def validate_config(self) -> None:
        """Validate all configuration parameters are within bounds"""
        try:
            for param, (min_val, max_val) in self.config_bounds.items():
                value = getattr(self, param)
                if not min_val <= value <= max_val:
                    raise ValueError(
                        f"{param} value {value} outside valid range [{min_val}, {max_val}]"
                    )
            
            if self.num_agents < 1:
                raise ValueError("num_agents must be positive")
            
            if self.input_size < 1:
                raise ValueError("input_size must be positive")

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def update_interactive(self) -> None:
        """Interactive configuration update with validation"""
        try:
            print("\nCurrent Agent Configuration:")
            for field in self.__dataclass_fields__:
                if not field.startswith('_'):
                    print(f"{field}: {getattr(self, field)}")

            if input("\nUpdate configuration? (y/n): ").lower() == 'y':
                for param, (min_val, max_val) in self.config_bounds.items():
                    current = getattr(self, param)
                    while True:
                        try:
                            new_val = input(
                                f"Enter {param} [{current}] "
                                f"(valid range: {min_val}-{max_val}): "
                            ) or current
                            
                            new_val = type(current)(new_val)
                            if min_val <= new_val <= max_val:
                                setattr(self, param, new_val)
                                break
                            else:
                                print(f"Value must be between {min_val} and {max_val}")
                        except ValueError:
                            print(f"Invalid input. Must be {type(current).__name__}")

                # Update non-bounded parameters
                self.num_agents = int(input(f"Enter num_agents [{self.num_agents}]: ") 
                                    or self.num_agents)
                self.input_size = int(input(f"Enter input_size [{self.input_size}]: ")
                                    or self.input_size)
                
                self.validate_config()
                logger.info("Configuration updated successfully")
                
                if input("Save configuration? (y/n): ").lower() == 'y':
                    self.save_config()

        except KeyboardInterrupt:
            logger.info("Configuration update cancelled by user")
            return
        except Exception as e:
            logger.error(f"Configuration update failed: {str(e)}")
            raise

    def save_config(self, filepath: Optional[str] = None) -> None:
        """Save configuration to JSON file"""
        try:
            if filepath is None:
                filepath = os.path.join(
                    self.backup_path, 
                    f"agent_config_{int(time.time())}.json"
                )
            
            config_dict = {
                field: getattr(self, field) 
                for field in self.__dataclass_fields__ 
                if not field.startswith('_')
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            logger.info(f"Loaded configuration from {filepath}")
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    @classmethod 
    def from_dict(cls, params: Dict, num_agents: int, input_size: int):
        try:
            config = cls(
                hidden_size=params['hidden_size'],
                memory_size=params['memory_size'],
                memory_vector_dim=params['memory_vector_dim'],
                num_heads=params['nhead'],
                num_layers=params['num_layers'],
                input_size=input_size,
                num_agents=num_agents
            )
            logger.info(f"Created AgentConfig with {num_agents} agents")
            return config
        except KeyError as e:
            logger.error(f"Missing required parameter: {e}")
            raise ValueError(f"Missing required parameter: {e}")
        except Exception as e:
            logger.error(f"Failed to create config: {e}")
            raise
