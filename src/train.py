import torch
from lightning.pytorch.loggers import WandbLogger
import lightning as L

from datasets import load_dataset
from transformers import AutoTokenizer

from safetensors.torch import load_file

import yaml
import click

from transformer import LightningTransformer
from dataset import LightningDataLoader
from utils.hf_upload import HFBucketRsync

@click.command()
@click.argument('train_config_file')
def main(train_config_file):
   with open(train_config_file, 'r') as f:
      config = yaml.safe_load(f)
      
      dataset_ckpt = config['dataset_ckpt']
      dataset_split = config['dataset_split']
      text_column = config['text_column']
      stream_dataset = bool(config['stream_dataset'])
      tokenizer_ckpt = config['tokenizer_ckpt']
      save_ckpt = config['save_ckpt']
      pretrain_ckpt = config['pretrain_ckpt']
      
      hf_bucket_name=config['hf_bucket_name']
      hf_bucket_save_dir=config['hf_bucket_save_dir']
      
      wandb_run_name = config['wandb_run_name']
      wandb_run_project = config['wandb_run_project']
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
      log_model=False,
      resume='allow',
      project=wandb_run_project,
      name=wandb_run_name
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
      iterations=(iterations // batch_acc)
   )
   
   model.load_state_dict(load_file("model.safetensors"))

   dataset = load_dataset(dataset_ckpt, split=dataset_split, streaming=stream_dataset)
   
   tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)

   dataloader = LightningDataLoader(
      tokenizer,
      dataset,
      text_column,
      batch_size,
      seq_len,
      num_workers,
   )

   trainer = L.Trainer(
      logger=wandb_logger,
      max_epochs=max_epochs,
      max_steps=iterations // batch_acc,
      limit_train_batches=iterations,
      precision=precision,
      gradient_clip_val=gradient_clip_val,
      accumulate_grad_batches=batch_acc,
      log_every_n_steps=log_every_n_steps,
      enable_checkpointing=True,
      devices=devices,
      strategy='auto',
      callbacks=[
         L.pytorch.callbacks.ModelCheckpoint(
            dirpath=save_ckpt, 
            every_n_train_steps=save_every_n_train_steps, 
            save_top_k=save_top_k,
         ),
         L.pytorch.callbacks.LearningRateMonitor(
            logging_interval='step'
         ),
         # HFBucketRsync(
         #    local_save_dir=save_ckpt,
         #    bucket_name=hf_bucket_name,
         #    bucket_save_dir=hf_bucket_save_dir,
         #    every_n_train_steps=save_every_n_train_steps+1
         # )
      ],
   )
    
   if pretrain_ckpt is not None: 
      trainer.fit(model, datamodule=dataloader, ckpt_path=pretrain_ckpt) # doesnt work
   else: 
      trainer.fit(model, datamodule=dataloader)
      
if __name__ == '__main__':
     main()