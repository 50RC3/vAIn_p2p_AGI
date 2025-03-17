import torch
import torch.nn as nn

class vAInTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4
            ),
            num_layers=config.num_layers
        )
        
    def forward(self, x):
        return self.encoder(x)
