import torch
import torch.nn as nn
import torch.nn.functinoal as F

class SwiGLU(nn.Module):
     def __init__(self, embed_dims, exp_factor):
          super().__init__()
          
          self.proj_up = nn.Linear(embed_dims, embed_dims*exp_factor)
          self.proj_down = nn.Linear(embed_dims*exp_factor, embed_dims)
          
     def forward(self, x):
          a, b = self.proj_up(x).chunk(2, dim=-1)
          y = F.SiLU(a) * b
          return self.proj_down(y)