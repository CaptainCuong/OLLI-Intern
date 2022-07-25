import torch
from torch.utils.data import Dataset


class MyData(Dataset):
  def __init__(self, data, pos_tag, label):
      assert len(data) == len(label)
      assert len(pos_tag) == len(data)
      self.pos_tag = pos_tag
      self.data = data
      self.label = label

  def __getitem__(self, index):
      return (self.data[index], self.pos_tag[index], self.label[index])

  def __len__(self):
      return len(self.data)


def create_dataset(data, tag, label):
  return MyData(data, tag, label)
