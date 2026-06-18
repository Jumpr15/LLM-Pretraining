import torch
from lightning.pytorch.loggers import WandbLogger
import lightning as L

import yaml

from nets.transformer import LightningTransformer
from data_module.dataset import LightningDataLoader

def main():
     with open('train_config.yaml', 'r') as f:
          config = yaml.safe_load(f)
          
          dataset_ckpt = config['dataset_ckpt']
          tokenizer_ckpt = config['tokenizer_ckpt']
          save_ckpt = config['save_ckpt']
          pretrain_ckpt = config['pretrain_ckpt']
          
          run_id = config['run_id']
          save_every_n_train_steps = int(config['save_every_n_train_steps'])
          save_top_k = int(config['save_top_k'])
          log_every_n_steps = int(config['log_every_n_steps'])
          
          precision = config['precision']
          gradient_clip_val = float(config['gradient_clip_val'])
          devices = int(config['devices'])
          
          batch_size = int(config['batch_size'])
          batch_acc = int(config['batch_acc'])
          lr = float(config['lr'])
          iterations = int(config['iterations'])
          max_epochs= int(config['max_epochs'])
          num_workers = int(config['num_workers'])
          
          seq_len = int(config['seq_len'])
          embed_dims = int(config['embed_dims'])
          head_size = int(config['head_size'])
          num_heads = int(config['num_heads'])
          block_num = int(config['block_num'])
          vocab_size = int(config['vocab_size'])
          
     wandb_logger = WandbLogger(
         log_model="all",
         resume="allow",
         id=run_id
     )

     model = LightningTransformer(
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dims=embed_dims,
        head_size=head_size,
        num_heads=num_heads,
        block_num=block_num,
        vocab_size=vocab_size,
        lr=lr,
        iterations=iterations,
     )
     
     model = torch.compile(model)
     
     dataloader = LightningDataLoader(
      dataset_ckpt,
      tokenizer_ckpt,
      batch_size,
      seq_len,
      num_workers
     )

     trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        limit_train_batches=iterations,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=batch_acc,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=True,
        devices=devices,
        strategy="auto",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
               dirpath=save_ckpt, every_n_train_steps=save_every_n_train_steps, save_top_k=save_top_k
            )
        ],
     )
    
     if pretrain_ckpt is not None: 
      trainer.fit(model, datamodule=dataloader, ckpt_path=pretrain_ckpt)
     else: 
      trainer.fit(model, datamodule=dataloader)
if __name__ == '__main__':
     main()