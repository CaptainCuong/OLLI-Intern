from torch.utils.data import Dataset
import torch

class MyData(Dataset):
  def __init__(self, data, label):
      assert len(data) == len(label)
      self.data = data
      self.label = label

  def __getitem__(self, index):
      return (self.data[index], self.label[index])

  def __len__(self):
      return len(self.data)


def create_dataset(data, label):
  return MyData(data, label)
