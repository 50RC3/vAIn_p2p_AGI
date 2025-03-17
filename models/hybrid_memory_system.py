import torch
import torch.nn as nn
from models.dnc.dnc_controller import DNCController
import logging

logger = logging.getLogger(__name__)

class HybridMemorySystemError(Exception):
    pass

class HybridMemorySystem(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, memory_size: int, 
                 memory_vector_dim: int, nhead: int, num_layers: int):
        try:
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # Create DNC controller
            self.dnc_controller = DNCController({
                'input_size': input_size,
                'hidden_size': hidden_size,
                'memory_size': memory_size,
                'memory_vector_dim': memory_vector_dim,
                'num_heads': nhead,
                'num_layers': num_layers
            })
            
            # Input projection (optional, can be used if input dimensions need adjustment)
            self.input_projection = nn.Linear(input_size, hidden_size)
            logger.info(f"Initialized HybridMemorySystem with memory size {memory_size}")
        except Exception as e:
            logger.error(f"Failed to initialize HybridMemorySystem: {str(e)}")
            raise HybridMemorySystemError(f"Initialization failed: {str(e)}")
        
    def forward(self, x):
        try:
            # Optional input projection
            if hasattr(self, 'input_projection'):
                x = self.input_projection(x)
                
            # Process through DNC
            transformer_out, read_vector = self.dnc_controller(x)
            return transformer_out, read_vector
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise HybridMemorySystemError(f"Forward pass failed: {str(e)}")

class HybridMemorySystem(nn.Module):
    def __init__(self, input_size, memory_size, memory_vector_dim):
        super().__init__()
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.memory = nn.Parameter(torch.randn(memory_size, memory_vector_dim))
        
        self.controller = nn.LSTM(input_size, memory_vector_dim)
        self.write_gate = nn.Linear(memory_vector_dim, 1)
        self.read_gate = nn.Linear(memory_vector_dim, 1)
        
    def forward(self, x):
        controller_output, _ = self.controller(x)
        write_weights = torch.softmax(self.write_gate(controller_output), dim=1)
        read_weights = torch.softmax(self.read_gate(controller_output), dim=1)
        
        # Memory operations
        memory_output = torch.matmul(read_weights, self.memory)
        return memory_output, self.memory
