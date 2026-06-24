import torch
import torch.nn as nn
import lightning as L
import importlib.util 

from torch.optim import AdamW, SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from huggingface_hub import PyTorchModelHubMixin

if importlib.util.find_spec('liger-kernel'):
    import liger_kernel.transformers as liger

from nets.attention_block import Block

class LightningTransformer(L.LightningModule, PyTorchModelHubMixin):
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
        tie_weights=False
    ):
        super().__init__()
        self.save_hyperparameters()
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
        
        # Set both layers to same weights if using weight tying (Torch auto-transposes)
        if tie_weights:
            self.token_embed.weight = self.embed_proj.weight
        
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
        
        wsd_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, stable_scheduler, cosine_decay_scheduler],
            milestones=[self.warmup_steps, self.iterations * (1 - self.decay_ratio)]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": wsd_scheduler, "interval": "step"},
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

        return unembed_out

    def generate(self, input_tokens, max_tokens):
        for _ in range(max_tokens):
            last_seq = input_tokens[:, -self.seq_len :]
            logits = self(last_seq)
            logits = logits[:, -1, :]
            probs = self.softmax(logits)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, next_tok), dim=1)
        return input_tokens