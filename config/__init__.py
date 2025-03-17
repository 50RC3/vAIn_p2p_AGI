from dataclasses import dataclass
from .blockchain_config import BlockchainConfig
from .training_config import TrainingConfig
from .network_config import NetworkConfig

@dataclass
class Config:
    blockchain: BlockchainConfig
    training: TrainingConfig
    network: NetworkConfig
    
    # Add missing config validation
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")

    @classmethod
    def load(cls):
        return cls(
            blockchain=BlockchainConfig.from_env(),
            training=TrainingConfig.from_env(),
            network=NetworkConfig.from_env()
        )

__all__ = ['Config']
