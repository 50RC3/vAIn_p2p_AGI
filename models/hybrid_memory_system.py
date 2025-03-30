"""Hybrid memory system combining neural and symbolic memory approaches."""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import enum
from dataclasses import dataclass

# Import DNCController lazily to avoid circular imports
def get_dnc_controller():
    from .dnc.dnc_controller import DNCController
    return DNCController

# Import MemoryManager lazily to avoid circular imports
def get_memory_manager():
    from memory.memory_manager import MemoryManager
    return MemoryManager

logger = logging.getLogger(__name__)

class MemoryAccessType(enum.Enum):
    """Types of memory access patterns"""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"

@dataclass
class MemoryOperation:
    """Represents a memory operation request"""
    operation_type: MemoryAccessType
    key: str
    value: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

class HybridMemorySystem(nn.Module):
    """A memory system that combines neural and symbolic approaches"""
    
    def __init__(self, 
                 input_size: int = 512,
                 hidden_size: int = 256,
                 memory_size: int = 128,
                 word_size: int = 64,
                 num_heads: int = 4,
                 symbolic_memory_size: int = 10000):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_heads = num_heads
        self.symbolic_memory_size = symbolic_memory_size
        
        # Use lazy loading to avoid circular imports
        DNCController = get_dnc_controller()
        
        # Initialize neural memory (DNC)
        self.neural_memory = DNCController({
            'input_size': input_size,
            'hidden_size': hidden_size,
            'memory_size': memory_size,
            'word_size': word_size,
            'num_heads': num_heads
        })
        
        # Initialize symbolic memory store
        MemoryManager = get_memory_manager()
        self.symbolic_memory = MemoryManager()
        
        # Memory router - maps input to appropriate memory system
        self.memory_router = nn.Linear(input_size, 2)  # 2 outputs: [neural_weight, symbolic_weight]
        
        # Memory integration - combines outputs from both memory systems
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Statistics tracking
        self.access_stats = {
            'neural': {'read': 0, 'write': 0},
            'symbolic': {'read': 0, 'write': 0}
        }
        
        # Initialize gateway
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the hybrid memory system"""
        # Determine memory routing weights
        routing_weights = torch.softmax(self.memory_router(x), dim=-1)
        neural_weight = routing_weights[:, 0].unsqueeze(1)
        symbolic_weight = routing_weights[:, 1].unsqueeze(1)
        
        # Process through neural memory (DNC)
        neural_output, _ = self.neural_memory(x)
        
        # Process through symbolic memory
        # Convert tensor to key for symbolic lookup
        batch_size = x.size(0)
        symbolic_outputs = []
        
        for i in range(batch_size):
            # Generate key from input tensor
            key = self._tensor_to_key(x[i])
            
            # Try to retrieve from symbolic memory
            try:
                value = self.symbolic_memory.get(key)
                if value is None:
                    # If not found, create a default tensor
                    value = torch.zeros(self.hidden_size, device=x.device)
                    self.access_stats['symbolic']['write'] += 1
                else:
                    self.access_stats['symbolic']['read'] += 1
            except Exception as e:
                logger.error(f"Symbolic memory error: {e}")
                value = torch.zeros(self.hidden_size, device=x.device)
            
            symbolic_outputs.append(value)
        
        # Stack outputs
        symbolic_output = torch.stack(symbolic_outputs)
        
        # Calculate gate value for dynamic integration
        gate_value = self.gate(x)
        
        # Combine outputs using dynamic gating
        combined_output = (gate_value * neural_output + 
                          (1 - gate_value) * symbolic_output)
        
        return self.integration_layer(combined_output)
    
    def _tensor_to_key(self, tensor: torch.Tensor) -> str:
        """Convert a tensor to a string key for symbolic memory"""
        # Simple hashing approach - can be improved
        import hashlib
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    async def store_memory(self, key: str, value: torch.Tensor, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a value in symbolic memory with metadata"""
        try:
            await self.symbolic_memory.store(key, value, metadata)
            self.access_stats['symbolic']['write'] += 1
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    async def retrieve_memory(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a value from symbolic memory by key"""
        try:
            value = await self.symbolic_memory.retrieve(key)
            if value is not None:
                self.access_stats['symbolic']['read'] += 1
            return value
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return None
