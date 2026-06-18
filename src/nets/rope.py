import torch
import liger_kernel.transformers as liger
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaConfig
)

class RoPE():
  def __init__(self, seq_len, num_heads, head_size):
    config = LlamaConfig(
      hidden_size=num_heads * head_size, # Total dimension of the model's embeddings
      num_attention_heads=num_heads,    # Number of attention heads
      num_key_value_heads=num_heads,   # Number of key/value heads (for GQA/MQA)
      max_position_embeddings=seq_len,    # Maximum sequence length for which embeddings are cached
      vocab_size=6767,                    # A placeholder vocab_size if not directly relevant
    )
    
    self.rotary_emb = LlamaRotaryEmbedding(config)

    self.pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

  def forward(self, q, k):
    cos, sin = self.rotary_emb(k, self.pos_ids)
    tt_q, tt_k = liger.liger_rotary_pos_emb(q, k, cos, sin)
    return tt_q, tt_k