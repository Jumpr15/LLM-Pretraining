import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaConfig
)

import importlib.util
if importlib.util.find_spec('liger_kernel'):
    import liger_kernel.transformers as liger

class RoPE(nn.Module):
  def __init__(self, seq_len, num_heads, head_size, use_liger, base=10000):
    super().__init__()

    self.use_liger = use_liger

    if self.use_liger:
      config = LlamaConfig(
        hidden_size=num_heads * head_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=seq_len,
        vocab_size=6767,
      )
      self.rotary_emb = LlamaRotaryEmbedding(config)

    else:
      self.base = base
      self.seq_len = seq_len
      self.dim = head_size

      self.build_cache()

  def build_cache(self):
    seq_idx = torch.arange(self.seq_len).float()
    theta = self.base ** ((-2/self.dim)*(torch.arange(0, self.dim/2).float()))

    idx_theta = seq_idx.unsqueeze(dim=1) @ theta.unsqueeze(dim=0)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

    sin_cached = idx_theta2.sin()[None, None, :, :]
    cos_cached = idx_theta2.cos()[None, None, :, :]
    
    self.register_buffer('sin_cached', sin_cached)
    self.register_buffer('cos_cached', cos_cached)

  def get_neg(self, x):
    x_1 = x[:, :, :, self.dim//2:]
    x_2 = x[:, :, :, :self.dim//2]
    x_neg = torch.cat([-x_1, x_2], dim=-1)
    return x_neg

  def forward(self, q, k):
    batch_size, seq_len = q.shape[0], q.shape[1]
    # position_ids must be (batch_size, seq_len)
    if self.use_liger:
      pos_ids = torch.arange(seq_len, dtype=torch.long, device=q.device).unsqueeze(0).expand(batch_size, -1)
      cos, sin = self.rotary_emb(k, pos_ids)
      q_rope, k_rope = liger.liger_rotary_pos_emb(q, k, cos, sin)
    else:
      q_rope = q * self.cos_cached + self.get_neg(q) * self.sin_cached
      k_rope = k * self.cos_cached + self.get_neg(k) * self.sin_cached      
    return q_rope, k_rope