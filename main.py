from utils import *
from data import *
from model import *
import numpy as np
import pandas as pd
import torch

data_file = pd.read_csv('dataset.csv')
snts = data_file['Sentences'].tolist()
labels = data_file['Label'].tolist()

snts, labels = word2vec(snts, labels)

seq_len = 50
snts = nml_sentence(seq_len, snts)
labels = nml_sentence(seq_len, labels)
print(snts)
print(labels)

dataset = create_dataset(torch.from_numpy(snts), torch.from_numpy(labels))
loader = create_loader(dataset)
