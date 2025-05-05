import torch
import torch.nn as nn

class vAInTransformer(nn.Module):
    """
    Custom transformer model for vAIn system
    """
    def __init__(self, dim, depth, heads, dim_head):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        
        # Basic transformer components
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dim_head // heads),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(),
                    nn.Linear(4 * dim, dim)
                )
            ]))
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x_norm = norm1(x)
            x = x + attn(x_norm, x_norm, x_norm)[0]
            x = x + ff(norm2(x))
        
        return self.final_norm(x)