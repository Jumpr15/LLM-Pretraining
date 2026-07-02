import lightning as L

from torch.utils.data import IterableDataset, DataLoader

class HFStreamingDataset(IterableDataset):
  def __init__(self, tokenizer, dataset, text_column, seq_len):
    super().__init__()
    self.seq_len = seq_len
    self.tokenizer = tokenizer
    self.dataset = dataset
    self.text_column = text_column

  def __iter__(self):
    for seq in self.dataset:
      text_seq = seq[self.text_column]
      enc = self.tokenizer(
        text_seq,
        max_length=self.seq_len + 1,
        truncation=True,
        return_tensors='pt'
      )
      enc_seq = enc['input_ids'].squeeze(0)
      
      if len(enc_seq) < self.seq_len + 1:
        continue
      
      yield enc_seq[:-1], enc_seq[1:]
  
class LightningDataLoader(L.LightningDataModule):
  def __init__(self, tokenizer, dataset, text_column, batch_size, seq_len, num_workers):
    super().__init__()
    self.tokenizer = tokenizer
    self.dataset = dataset
    self.text_column = text_column
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.num_workers = num_workers
    
  def setup(self, stage=None):
    self.train_dataset = HFStreamingDataset(self.tokenizer, self.dataset, self.text_column, self.seq_len)
    
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      pin_memory=True
    )
  