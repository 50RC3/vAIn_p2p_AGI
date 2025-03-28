import torch
import torch.nn as nn
from models.dnc.dnc_controller import DNCController
import logging
from typing import Tuple, Any, Optional
from memory.memory_manager import MemoryManager
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INTERACTION_TIMEOUTS = {
    "batch": 30,
    "emergency": 10
}

class HybridMemorySystemError(Exception):
    pass

class InteractionLevel(Enum):
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"

@dataclass
class InteractiveConfig:
    timeout: int
    persistent_state: bool
    safe_mode: bool
    max_cleanup_wait: int

    async def __aenter__(self) -> 'InteractiveSession':
        return self

    async def __aexit__(self, exc_type: Optional[Any], exc_val: Optional[Any], exc_tb: Optional[Any]) -> None:
        pass

    async def get_confirmation(self, message: str, timeout: int) -> bool:
        logger.info("Confirmation requested: %s (timeout: %d)", message, timeout)
        # Implement your confirmation logic here
        return True

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
            self.memory_manager = MemoryManager(max_cache_size=1024*1024*1024, interactive=interactive)
            logger.info("Initialized HybridMemorySystem with memory size %d", memory_size)
        except Exception as e:
            logger.error("Failed to initialize HybridMemorySystem: %s", str(e))
            raise HybridMemorySystemError(f"Initialization failed: {str(e)}") from e
        
    def forward(self, x):
        try:
            # Optional input projection
            if hasattr(self, 'input_projection'):
                x = self.input_projection(x)
                
            # Process through DNC
            transformer_out, read_vector = self.dnc_controller(x)
            return transformer_out, read_vector
        except Exception as e:
            logger.error("Forward pass failed: %s", str(e))
            raise HybridMemorySystemError(f"Forward pass failed: {str(e)}") from e

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
                        logger.error("Interactive forward pass failed: %s", str(e))

                # Forward pass with monitoring
                return await self._monitored_forward(x)

        except Exception as e:
            logger.error(f"Interactive forward pass failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def __aenter__(self) -> 'HybridMemorySystem':
        if self.interactive:
            await self.memory_manager.register_memory_system("hybrid_system", self)
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)

    async def _check_memory_usage(self) -> bool:
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

            # Cache input tensor if large
            if x.numel() * x.element_size() > 1024*1024:  # 1MB
                await self.memory_manager.cache_tensor_interactive("input", x)

            # Perform forward pass
            transformer_out, read_vector = self.forward(x)

            # Cache large intermediate results
            if transformer_out.numel() * transformer_out.element_size() > 1024*1024:
                await self.memory_manager.cache_tensor_interactive("transformer_out", transformer_out)

            # Log memory usage
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated()
                logger.debug("Memory usage: %.2fMB", (final_mem - initial_mem) / 1024**2)

            return transformer_out, read_vector

        except Exception as e:
            logger.error(f"Monitored forward pass failed: {str(e)}")
            raise
