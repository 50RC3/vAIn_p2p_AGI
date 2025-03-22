import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict
from torch.nn import TransformerEncoderLayer, LayerNorm
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS
import math

logger = logging.getLogger(__name__)

class SimpleNNError(Exception):
    pass

class AdvancedNN(nn.Module):
    def __init__(self, config: Dict):
        try:
            super().__init__()
            # Extract configuration
            self.input_dim = config.get('input_dim', 784)
            self.hidden_dim = config.get('hidden_dim', 256)
            self.output_dim = config.get('output_dim', 10)
            self.num_layers = config.get('num_layers', 4)
            self.nhead = config.get('nhead', 8)
            self.dropout = config.get('dropout', 0.1)
            self.interactive = config.get('interactive_mode', True)
            self.session = None
            self._interrupt_requested = False
            self.progress_file = "model_progress.json"
            
            # Input projection
            self.input_projection = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                LayerNorm(self.hidden_dim)
            )
            
            # Transformer encoder layers with residual connections
            self.transformer_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=4*self.hidden_dim,
                    dropout=self.dropout,
                    activation='gelu',
                    batch_first=True
                ) for _ in range(self.num_layers)
            ])
            
            # Residual dense layers
            self.residual_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    LayerNorm(self.hidden_dim),
                    nn.Dropout(self.dropout)
                ) for _ in range(2)
            ])
            
            # Output projection
            self.output_projection = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                LayerNorm(self.hidden_dim // 2),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
            
            # Initialize weights
            self._init_weights()
            
            logger.info(f"Initialized AdvancedNN with {self.num_layers} transformer layers")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdvancedNN: {str(e)}")
            raise SimpleNNError(f"Model initialization failed: {str(e)}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            # Input shape handling
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
                
            # Initial projection
            hidden = self.input_projection(x)
            
            # Transformer layers with residual connections
            attention_weights = []
            for transformer in self.transformer_layers:
                transformer_out = transformer(hidden)
                attention_weights.append(transformer.self_attn.attention_weights)
                hidden = hidden + transformer_out
            
            # Residual dense layers
            for layer in self.residual_layers:
                residual = layer(hidden)
                hidden = hidden + residual
            
            # Global average pooling over sequence dimension
            hidden = torch.mean(hidden, dim=1)
            
            # Output projection
            output = self.output_projection(hidden)
            
            # Return output and attention weights
            return output, attention_weights
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise SimpleNNError(f"Forward pass failed: {str(e)}")

    async def forward_interactive(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Interactive forward pass with progress tracking and resource monitoring"""
        if not self.interactive:
            return self.forward(x)

        try:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True
                )
            )

            async with self.session:
                # Check resources
                if not await self._check_resources():
                    raise RuntimeError("Insufficient resources")

                # Monitor GPU memory
                if torch.cuda.is_available():
                    initial_mem = torch.cuda.memory_allocated()

                output, attention = await self._forward_with_progress(x)

                # Log memory usage
                if torch.cuda.is_available():
                    final_mem = torch.cuda.memory_allocated()
                    logger.debug(f"GPU Memory delta: {(final_mem - initial_mem) / 1024**2:.2f}MB")

                return output, attention

        except Exception as e:
            logger.error(f"Interactive forward pass failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _check_resources(self) -> bool:
        """Check system resources"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                if self.interactive and self.session:
                    proceed = await self.session.get_confirmation(
                        "High memory usage detected. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
                    return proceed
                return False
            return True
        except Exception as e:
            logger.warning(f"Resource check failed: {str(e)}")
            return True

    async def _forward_with_progress(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with progress tracking"""
        try:
            # ...existing forward pass code with progress tracking...
            return output, attention_weights
        except Exception as e:
            logger.error(f"Forward pass with progress failed: {str(e)}")
            raise

    def save(self, path: str) -> None:
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise SimpleNNError(f"Model save failed: {str(e)}")

    def load(self, path: str) -> None:
        try:
            self.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise SimpleNNError(f"Model load failed: {str(e)}")
