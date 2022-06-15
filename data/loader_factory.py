from torch.utils.data import DataLoader


def create_loader(dataset):
	return DataLoader(dataset, drop_last=True)