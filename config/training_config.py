from dataclasses import dataclass, field
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    hidden_size: int
    num_layers: int
    num_rounds: int
    min_clients: int
    clients_per_round: int
    
    # DNC parameters
    input_size: int = int(os.getenv('INPUT_SIZE', 256))
    memory_size: int = int(os.getenv('MEMORY_SIZE', 128))
    memory_vector_dim: int = int(os.getenv('MEMORY_VECTOR_DIM', 64))
    num_heads: int = int(os.getenv('NUM_HEADS', 4))

    # Interactive settings
    interactive: bool = True
    progress_bar: bool = True
    log_interval: int = 10
    save_interval: int = 100
    
    # Resource management 
    max_memory_usage: float = 0.9  # 90% of available memory
    max_cpu_usage: float = 0.8     # 80% of CPU
    device: str = 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu'
    
    # Recovery and checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: str = 'checkpoints'
    backup_interval: int = 1800  # 30 minutes
    
    # Monitoring
    metrics_enabled: bool = True
    alert_threshold: float = 0.1  # Alert if metrics drop by >10%
    
    # Error handling
    max_retries: int = 3
    timeout: int = 3600  # 1 hour timeout
    
    # Resource limits per client
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        'max_batch_memory': '8G',
        'max_concurrent_tasks': 4,
        'min_free_memory': '2G',
        'timeout_per_round': 600
    })

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_directories()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.min_clients > self.clients_per_round:
            raise ValueError("min_clients cannot exceed clients_per_round")
            
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        if self.enable_checkpointing:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def update_from_env(self) -> None:
        """Update config from environment variables"""
        try:
            for key, value in os.environ.items():
                if hasattr(self, key.lower()):
                    attr_type = type(getattr(self, key.lower()))
                    setattr(self, key.lower(), attr_type(value))
        except Exception as e:
            logger.error(f"Error updating config from env: {str(e)}")
            raise

    def get_resource_limit(self, resource: str) -> Optional[Any]:
        """Get resource limit with validation"""
        return self.resource_limits.get(resource)

    @classmethod
    def from_env(cls):
        """Create config from environment variables with validation"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            config = cls(
                batch_size=int(os.getenv('BATCH_SIZE', 32)),
                learning_rate=float(os.getenv('LEARNING_RATE', 0.001)),
                num_epochs=int(os.getenv('NUM_EPOCHS', 100)),
                hidden_size=int(os.getenv('HIDDEN_SIZE', 256)),
                num_layers=int(os.getenv('NUM_LAYERS', 4)),
                num_rounds=int(os.getenv('NUM_ROUNDS', 50)),
                min_clients=int(os.getenv('MIN_CLIENTS', 3)),
                clients_per_round=int(os.getenv('CLIENTS_PER_ROUND', 10)),
                input_size=int(os.getenv('INPUT_SIZE', 256)),
                memory_size=int(os.getenv('MEMORY_SIZE', 128)),
                memory_vector_dim=int(os.getenv('MEMORY_VECTOR_DIM', 64)),
                num_heads=int(os.getenv('NUM_HEADS', 4))
            )
            
            # Validate after creation
            config._validate_config()
            return config
            
        except Exception as e:
            logger.error(f"Failed to create config from env: {str(e)}")
            raise

    def update_interactive(self) -> None:
        """Interactive configuration update"""
        try:
            print("\nCurrent Training Configuration:")
            for field in self.__dataclass_fields__:
                if not field.startswith('_'):
                    print(f"{field}: {getattr(self, field)}")

            if input("\nUpdate configuration? (y/n): ").lower() == 'y':
                self.batch_size = int(input(f"Enter batch_size [{self.batch_size}]: ") or self.batch_size)
                self.learning_rate = float(input(f"Enter learning_rate [{self.learning_rate}]: ") or self.learning_rate)
                self.num_epochs = int(input(f"Enter num_epochs [{self.num_epochs}]: ") or self.num_epochs)
                self.clients_per_round = int(input(f"Enter clients_per_round [{self.clients_per_round}]: ") or self.clients_per_round)
                self.min_clients = int(input(f"Enter min_clients [{self.min_clients}]: ") or self.min_clients)
                
                self._validate_config()
                logger.info("Training configuration updated successfully")

        except KeyboardInterrupt:
            logger.info("Configuration update cancelled by user")
            return
        except Exception as e:
            logger.error(f"Configuration update failed: {str(e)}")
            raise
