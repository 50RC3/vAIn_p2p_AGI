import torch
import torch.optim as optim
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from models.hybrid_memory_system import HybridMemorySystem
from .meta_reptile import MetaReptile

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    input_size: int
    hidden_size: int
    memory_size: int
    memory_vector_dim: int
    num_heads: int
    num_layers: int
    inner_learning_rate: float
    meta_learning_rate: float
    learning_rate: float
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if any(v <= 0 for v in [self.input_size, self.hidden_size, 
               self.memory_size, self.memory_vector_dim, 
               self.num_heads, self.num_layers]):
            raise ValueError("All size/dimension parameters must be positive")
        if any(not 0 < v < 1 for v in [self.inner_learning_rate,
               self.meta_learning_rate, self.learning_rate]):
            raise ValueError("Learning rates must be between 0 and 1")

class Agent:
    def __init__(self, config: AgentConfig):
        """Initialize agent with validated configuration"""
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        
        try:
            self.model = HybridMemorySystem(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                memory_vector_dim=config.memory_vector_dim,
                nhead=config.num_heads,
                num_layers=config.num_layers
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize HybridMemorySystem: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
            
        try:
            self.meta_reptile = MetaReptile(
                self.model,
                inner_lr=config.inner_learning_rate,
                meta_lr=config.meta_learning_rate
            )
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=config.learning_rate)
        except Exception as e:
            logger.error(f"Failed to initialize training components: {e}")
            raise RuntimeError(f"Training setup failed: {e}")

    def local_update(self, x: torch.Tensor, y: torch.Tensor, 
                    meta_steps: int) -> Tuple[float, dict]:
        """Perform local model update with monitoring"""
        if meta_steps <= 0:
            raise ValueError("meta_steps must be positive")
            
        if not x.size(0) == y.size(0):
            raise ValueError("Batch sizes of x and y must match")
            
        try:
            x = x.to(self.device)
            y = y.to(self.device)
            
            metrics = {
                'memory_used': torch.cuda.memory_allocated() 
                              if torch.cuda.is_available() else 0
            }
            
            loss = 0
            for step in range(meta_steps):
                try:
                    loss = self.meta_reptile.adapt_to_task(x, y)
                    metrics[f'step_{step}_loss'] = float(loss)
                except Exception as e:
                    logger.error(f"Error at meta-step {step}: {e}")
                    raise
                    
            metrics['final_loss'] = float(loss)
            return loss, metrics
            
        except Exception as e:
            logger.error(f"Local update failed: {e}")
            raise RuntimeError(f"Error during local update: {e}")
        
    def cleanup(self):
        """Release resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
