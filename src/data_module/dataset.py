from datasets import load_dataset
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader

class HFStreamingDataset(Dataset):
  def __init__(self, dataset_ckpt, tokenizer_ckpt, batch_size) -> None:
    super().__init__()
    self.dataset = load_dataset(dataset_ckpt, split='train', streaming=True)
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    self.batch_size = batch_size

  def __len__(self):
    return 99999999

  def __getitem__(self, idx):
    batch_seq = []

    for i in range(self.batch_size):
      text_seq = next(iter(self.dataset['text']))
      enc_seq = self.tokenizer.encode(text_seq, return_tensors='pt')
      batch_seq.append(enc_seq)

    batch_seq = torch.cat(batch_seq, dim=0)
    batch_seq = batch_seq.squeeze(dim=0)
    return batch_seq