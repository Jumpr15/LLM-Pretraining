import torch
import torch.nn as nn
import lightning as L
import liger_kernel.transformers as liger

from torchao.optim import AdamW8bit
from transformers import get_cosine_schedule_with_warmup
from huggingface_hub import PyTorchModelHubMixin

from llm_updated.src.nets.attention_block import Block

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.head_size = head_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.lr = lr
        self.iterations = iterations

        self.softmax = liger.LigerSoftmax()
        self.cross_entropy = liger.LigerCrossEntropyLoss()

        self.token_embed = nn.Embedding(vocab_size, embed_dims)
        self.rms_Norm_embed = liger.LigerRMSNorm(embed_dims)
        self.embed_proj = nn.Linear(embed_dims, vocab_size)
        self.block_list = nn.ModuleList(
            [Block(seq_len, embed_dims, head_size, num_heads) for _ in range(block_num)]
        )

    def configure_optimizers(self):
        optimizer = AdamW8bit(self.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            self.iterations*0.05,
            self.iterations*0.95
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
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