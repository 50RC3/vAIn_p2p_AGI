import torch
import torch.nn as nn
from models.dnc.dnc_controller import DNCController
import logging
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class HybridMemorySystemError(Exception):
    pass

class HybridMemorySystem(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, memory_size: int, 
                 memory_vector_dim: int, nhead: int, num_layers: int,
                 interactive: bool = True):
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
            self.interactive = interactive
            self.session = None
            self._interrupt_requested = False
            self.progress_file = "memory_progress.json"
            
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

    async def forward_interactive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interactive forward pass with memory monitoring"""
        if not self.interactive:
            return self.forward(x)

        try:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True,
                    max_cleanup_wait=30
                )
            )

            async with self.session:
                # Monitor memory usage
                if not await self._check_memory_usage():
                    if await self.session.get_confirmation(
                        "High memory usage detected. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    ):
                        logger.warning("Proceeding despite high memory usage")
                    else:
                        raise RuntimeError("Operation cancelled due to high memory usage")

                # Forward pass with monitoring
                return await self._monitored_forward(x)

        except Exception as e:
            logger.error(f"Interactive forward pass failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _check_memory_usage(self) -> bool:
        """Check system memory usage"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.percent < 90
        except Exception as e:
            logger.warning(f"Memory check failed: {str(e)}")
            return True

    async def _monitored_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with resource monitoring"""
        try:
            # Track initial memory
            if torch.cuda.is_available():
                initial_mem = torch.cuda.memory_allocated()

            # Perform forward pass
            transformer_out, read_vector = self.forward(x)

            # Log memory usage
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated()
                logger.debug(f"Memory usage: {(final_mem - initial_mem) / 1024**2:.2f}MB")

            return transformer_out, read_vector

        except Exception as e:
            logger.error(f"Monitored forward pass failed: {str(e)}")
            raise

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
