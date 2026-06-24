import torch.nn as nn

if importlib.util.find_spec('liger-kernel'):
    import liger_kernel.transformers as liger

from nets.attention_head import Attention_Head
from nets.swiglu_mlp import SwiGLUMLP_Config
from nets.swiglu import SwiGLU

class Block(nn.Module):
    def __init__(self, seq_len, embed_dims, head_size, num_heads, use_liger, exp_factor=3):
        super().__init__()
        self.embed_dims = embed_dims
        self.head_size = head_size

        if use_liger:
            self.rms_Norm1 = liger.LigerRMSNorm(embed_dims)
            self.rms_Norm2 = liger.LigerRMSNorm(embed_dims)
            
            config = SwiGLUMLP_Config(embed_dims, 'swish', exp_factor)
            self.FFN = liger.LigerSwiGLUMLP(config)
        
        else:
            self.rms_Norm1 = nn.RMSNorm(embed_dims)
            self.rms_Norm2 = nn.RMSNorm(embed_dims)
            
            self.FFN = SwiGLU(embed_dims, exp_factor)

        self.Attention_Head = Attention_Head(seq_len, embed_dims, head_size, num_heads, use_liger)

    def forward(self, logits, batch_size, seq_len):
        x = self.Attention_Head(self.rms_Norm1(logits), batch_size, seq_len)
        x = x + logits
        out = self.FFN(self.rms_Norm2(x))
        out = out + x
        return out