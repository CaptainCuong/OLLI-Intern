from utils import *
from data import *
from model import *
from utils import *
import numpy as np
import pandas as pd
import torch

data_file = pd.read_csv('dataset.csv')
snts = data_file['Sentences'].tolist()
labels = data_file['Label'].tolist()

snts, labels, vocab_size = word2vec(snts, labels)

seq_len = 50
snts = nml_sentence(seq_len, snts)
labels = nml_sentence(seq_len, labels)

dataset = create_dataset(torch.from_numpy(snts), torch.from_numpy(labels))
loader = create_loader(dataset)

output_size = n_entity = len(token)
embedding_dim = 128
hidden_dim = 256
n_layers = 2
model = NER_LSTMNet(vocab_size, n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=seq_len)
