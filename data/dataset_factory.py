from torch.utils.data import Dataset

class MyData(Dataset):
  def __init__(self, data, label):
      assert len(data) == len(label)
      self.data = data
      self.label = label

  def __getitem__(self, ind):
      return (self.data[ind], self.label[ind])

  def __len__(self):
      return len(data)


def create_dataset(data, label):
  return MyData(data, label)
