import torch
import torch.nn as nn

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

    def forward(self, x):
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
