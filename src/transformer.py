import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from torch.optim import AdamW

from huggingface_hub import PyTorchModelHubMixin
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

import lightning as L

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaConfig
)

import importlib.util 
if importlib.util.find_spec('liger_kernel'):
    import liger_kernel.transformers as liger
    
class WSD_Scheduler():
     def __init__(self, warmup_steps, iterations, optimizer, decay_ratio):
          
          self.warmup_steps = warmup_steps
          self.iterations = iterations
          self.decay_ratio = decay_ratio
          
          warmup_scheduler = LinearLR(
               optimizer,
               start_factor=0.1,
               end_factor=1.0,
               total_iters=self.warmup_steps
          )

          stable_scheduler = ConstantLR(
               optimizer,
               factor=1.0
          )

          cosine_decay_scheduler = CosineAnnealingLR(
               optimizer, 
               T_max=self.iterations*self.decay_ratio
          )

          self.wsd_scheduler = SequentialLR(
               optimizer,
               schedulers=[warmup_scheduler, stable_scheduler, cosine_decay_scheduler],
               milestones=[self.warmup_steps, self.iterations * (1 - self.decay_ratio)]
          )
          
     def get_scheduler(self):
          return self.wsd_scheduler

class SwiGLUMLP_Config():
    def __init__(
        self, 
        hidden_size: int, 
        hidden_act: int, 
        exp_factor: int,
    ):
      self.hidden_size = hidden_size
      self.intermediate_size = hidden_size*exp_factor
      self.hidden_act = hidden_act

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

class Attention_Head(nn.Module):
    def __init__(self, seq_len, embed_dims, head_size, num_heads, use_liger):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_size = head_size
        self.total_heads = head_size * num_heads

        self.q_proj = nn.Linear(embed_dims, self.total_heads)
        self.k_proj = nn.Linear(embed_dims, self.total_heads)
        self.v_proj = nn.Linear(embed_dims, self.total_heads)
        self.o_proj = nn.Linear(self.total_heads, embed_dims)
        self.pe = RoPE(seq_len, num_heads, head_size, use_liger)

    def forward(self, logits, batch_size, seq_len):
          q = self.q_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size)
          k = self.k_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size)

          q_pe, k_pe = self.pe.forward(q, k)

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

class LightningTransformer(L.LightningModule, PyTorchModelHubMixin, GenerationMixin):
    def __init__(
        self,
        batch_size,
        seq_len,
        embed_dims,
        head_size,
        num_heads,
        block_num,
        vocab_size,
        lr,
        iterations,
        warmup_steps=2000,
        decay_ratio=0.1,
        use_liger=False,
    ):
        super().__init__()
        self.save_hyperparameters() # Logs hyperparameters to WandB
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.head_size = head_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        self.block_list = nn.ModuleList(
            [Block(seq_len, embed_dims, head_size, num_heads, use_liger) for _ in range(block_num)]
        )

        self.lr = lr
        self.iterations = iterations
        self.warmup_steps = warmup_steps
        self.decay_ratio = decay_ratio
            
        self.token_embed = nn.Embedding(vocab_size, embed_dims)
        self.embed_proj = nn.Linear(embed_dims, vocab_size)
        
        # use Liger kernel if CUDA is available and LigerKernel is installed
        if use_liger: 
            self.softmax = liger.LigerSoftmax()
            self.cross_entropy = liger.LigerCrossEntropyLoss()
            self.rms_Norm_embed = liger.LigerRMSNorm(embed_dims)
           
        # fallback to Pytorch and Transformers 
        else: 
            self.softmax = nn.Softmax(dim=-1)
            self.cross_entropy = nn.CrossEntropyLoss()
            self.rms_Norm_embed = nn.RMSNorm(embed_dims)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, 
                mean=0.0, 
                std=0.02 * (self.embed_dims ** 0.5)
            )
        elif isinstance(module, nn.RMSNorm):
            torch.nn.init.ones_(module.weight)   
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        
        wsd_scheduler = WSD_Scheduler(self.warmup_steps, self.iterations, optimizer, self.decay_ratio)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": wsd_scheduler.get_scheduler(), "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, inputs, target=None):
        batch_size, seq_len = inputs.shape
        logits = self.token_embed(inputs)

        for block in self.block_list:
            logits = block(logits, batch_size, seq_len)

        unembed_out = self.embed_proj(self.rms_Norm_embed(logits))

        if target is not None:
            preds = unembed_out.view(batch_size * seq_len, -1)
            target = target.view(-1)

            loss_fn = self.cross_entropy(preds, target)
            return loss_fn

        return CausalLMOutput(logits=unembed_out)

    def generate(self, input_tokens, max_tokens):
        for _ in range(max_tokens):
            last_seq = input_tokens[:, -self.seq_len :]
            logits = self(last_seq)
            logits = logits[:, -1, :]
            probs = self.softmax(logits)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, next_tok), dim=1)
        return input_tokens