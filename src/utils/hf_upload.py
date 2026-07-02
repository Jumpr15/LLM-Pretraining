from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

from huggingface_hub import sync_bucket

class HFBucketRsync(L.pytorch.Callback):
  def __init__(
      self,
      local_save_dir: str,
      bucket_name: str,
      bucket_save_dir: str,
      every_n_train_steps: int,
  ):
    super().__init__()
    self.local_save_dir = local_save_dir
    self.bucket_name = bucket_name
    self.bucket_save_dir = bucket_save_dir
    self.every_n_train_steps = every_n_train_steps
    
  def on_train_batch_end(
      self,
      trainer: "L.Trainer",
      pl_module: "L.LightningModule",
      outputs: STEP_OUTPUT,
      batch: Any,
      batch_idx: int
  ) -> None:
    if trainer.global_step % self.every_n_train_steps == 0:
      sync_bucket(self.local_save_dir, f"hf://buckets/{self.bucket_name}/{self.bucket_save_dir}")