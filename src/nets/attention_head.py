import torch.nn as nn
import torch.nn.functional as F

from llm_updated.src.nets.rope import RoPE

class Attention_Head(nn.Module):
    def __init__(self, seq_len, embed_dims, head_size, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_size = head_size
        self.total_heads = head_size * num_heads

        self.q_proj = nn.Linear(embed_dims, self.total_heads)
        self.k_proj = nn.Linear(embed_dims, self.total_heads)
        self.v_proj = nn.Linear(embed_dims, self.total_heads)
        self.o_proj = nn.Linear(self.total_heads, embed_dims)
        self.pe = RoPE(seq_len, num_heads, head_size)

    def forward(self, logits, batch_size, seq_len):
          q = self.q_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size)
          k = self.k_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size)

          q_pe, k_pe = self.RoPE.forward(q, k)

          q_pe = q_pe.transpose(1, 2)
          k_pe = k_pe.transpose(1, 2)

          v = (
               self.v_proj(logits)
               .view(batch_size, seq_len, self.num_heads, self.head_size)
               .transpose(1, 2)
          )

          attention_out = F.scaled_dot_product_attention(q_pe, k_pe, v, is_causal=True)
          out = (
               attention_out.transpose(1, 2)
               .contiguous()
               .view(batch_size, seq_len, self.total_heads)
          )
          return self.o_proj(out)