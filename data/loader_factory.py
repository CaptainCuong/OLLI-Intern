from torch.utils.data import DataLoader

class MyLoader(DataLoader):
	def __init__(self, dataset, batch_size, drop_last):
		super().__init__(dataset, drop_last = drop_last, batch_size = batch_size)

	def __getitem__(self, index):
		return self.dataset[index]

def create_loader(dataset, batch_size, drop_last):
	return MyLoader(dataset, batch_size=batch_size, drop_last=True)