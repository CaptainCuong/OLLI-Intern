import pandas as pd

def create_data(data_root, train_size, test_percent = 0.2)
	'''
	Data is accomodated in a file with directory stored in data_root
	'''
	file = pd.csv('../dataset.csv')
	
	train_file = file[]
	test_file = file[-len(file)*test_percent: ]

	

