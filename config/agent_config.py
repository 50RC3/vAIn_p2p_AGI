from dataclasses import dataclass
from typing import Dict
import logging

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
