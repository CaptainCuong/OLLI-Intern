from torch.utils.data import Dataset

class MyData(Dataset):
	def __init__(self, data, label):
		super().__init__(data, label)


def create_dataset(data, label):
	return MyData(data, label)