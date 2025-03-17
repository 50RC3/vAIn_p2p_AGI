import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict
from torch.nn import TransformerEncoderLayer, LayerNorm
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
            raise SimpleNNError(f"Forward pass failed: {str(e)})

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
