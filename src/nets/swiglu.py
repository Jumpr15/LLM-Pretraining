import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self, 
        embed_dims: int, 
        exp_factor: int,
    ):
        super().__init__()

        self.up_proj = nn.Linear(embed_dims, embed_dims*exp_factor)
        self.gate_proj = nn.Linear(embed_dims, embed_dims*exp_factor)
        self.down_proj = nn.Linear(embed_dims*exp_factor, embed_dims)

    def forward(self, x):
        y = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(y)