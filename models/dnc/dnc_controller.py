import torch
import torch.nn as nn
import logging
import psutil
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class DNCController(nn.Module):
    def __init__(self, config):
        super(DNCController, self).__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        self.memory_vector_dim = config.memory_vector_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        # Transformer Encoder setup
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_size, 
            nhead=self.num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Initialize memory and attention weights as buffers (persistent)
        self.register_buffer('memory', torch.zeros(self.memory_size, self.memory_vector_dim))
        self.register_buffer('read_weights', torch.zeros(self.memory_size))
        self.register_buffer('write_weights', torch.zeros(self.memory_size))

        # Memory monitoring
        self.interactive = getattr(config, 'interactive', True)
        self.memory_threshold = getattr(config, 'memory_threshold', 0.9)
        self._session = None
        self._interrupt_requested = False

    async def __aenter__(self):
        """Context manager entry"""
        if self.interactive:
            self._session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True,
                    memory_threshold=self.memory_threshold
                )
            )
            await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
            self._session = None

    def _check_memory_usage(self):
        """Monitor memory usage"""
        memory_percent = psutil.Process().memory_percent()
        if memory_percent > self.memory_threshold * 100:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
            return False
        return True

    async def _safe_memory_update(self, erase_term, write_term):
        """Safe memory update with validation"""
        try:
            if torch.isnan(erase_term).any() or torch.isnan(write_term).any():
                raise ValueError("NaN values detected in memory update terms")

            if self.interactive and self._session:
                if not await self._session.get_confirmation("Proceed with memory update?"):
                    logger.info("Memory update cancelled by user")
                    return False

            new_memory = erase_term + write_term
            if not torch.isnan(new_memory).any():
                self.memory = new_memory
                return True
            else:
                logger.error("Invalid memory state after update")
                return False
        except Exception as e:
            logger.error(f"Memory update failed: {str(e)}")
            return False

    async def forward_interactive(self, x):
        """Interactive forward pass with safety checks"""
        try:
            if not self._check_memory_usage():
                if self.interactive and self._session:
                    proceed = await self._session.get_confirmation(
                        "High memory usage detected. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
                    if not proceed:
                        raise RuntimeError("Operation cancelled due to high memory usage")

            # Transformer processing
            transformer_out = self.transformer_encoder(x)
            read_vector = torch.matmul(self.read_weights, self.memory)
            write_vector = transformer_out[-1, :, :]

            # Safe memory update
            erase_term = self.memory * (1 - self.write_weights.unsqueeze(1))
            write_term = write_vector.unsqueeze(0) * self.write_weights.unsqueeze(1)
            
            if not await self._safe_memory_update(erase_term, write_term):
                logger.warning("Using previous memory state")

            return transformer_out, read_vector

        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def forward(self, x):
        """Standard forward pass for non-interactive use"""
        # Transformer processing
        transformer_out = self.transformer_encoder(x)
        
        # DNC read/write operations (simplified)
        read_vector = torch.matmul(self.read_weights, self.memory)
        write_vector = transformer_out[-1, :, :]  # Use last Transformer output
        
        # Update memory
        erase_term = self.memory * (1 - self.write_weights.unsqueeze(1))
        write_term = write_vector.unsqueeze(0) * self.write_weights.unsqueeze(1)
        self.memory = erase_term + write_term
        
        return transformer_out, read_vector
