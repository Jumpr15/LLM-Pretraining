import torch.nn as nn
import liger_kernel.transformers as liger

from llm_updated.src.nets.attention_head import Attention_Head
from llm_updated.src.nets.swiglu_mlp import SwiGLUMLP_Config

class Block(nn.Module):
    def __init__(self, seq_len, embed_dims, head_size, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.head_size = head_size

        self.rms_Norm1 = liger.LigerRMSNorm(embed_dims)
        self.rms_Norm2 = liger.LigerRMSNorm(embed_dims)

        self.Attention_Head = Attention_Head(seq_len, embed_dims, head_size, num_heads)

        config = SwiGLUMLP_Config(embed_dims, 'swish')
        self.FFN = liger.LigerSwiGLUMLP(config)

    def forward(self, logits, batch_size, seq_len):
        x = self.Attention_Head(self.rms_Norm1(logits), batch_size, seq_len)
        x = x + logits
        out = self.FFN(self.rms_Norm2(x))
        out = out + x
        return out