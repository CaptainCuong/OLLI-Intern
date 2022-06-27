import gensim
import numpy as np
import pandas as pd
import torch
from data import *
from gensim.models import KeyedVectors
from model import *
from utils import *

# snts, labels, vocab_size = word2vec(snts, labels)
# SEQ_LEN = 50
# snts = nml_sentence(SEQ_LEN, snts)
# labels = nml_label(SEQ_LEN, labels)

# BATCH_SIZE = 4
# dataset = create_dataset(torch.from_numpy(snts), torch.from_numpy(labels))
# loader = create_loader(dataset, BATCH_SIZE, True)

# output_size = n_entity = len(token)
# embedding_dim = 128
# hidden_dim = 256
# n_layers = 2
# model = NER_LSTMNet(vocab_size, n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN)
# model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))

# print(model(loader[0].unsqueeze(0)))
# model = torch.load('wiki.vi.model.bin')

data_file = pd.read_csv('dataset.csv')
snts = data_file['Sentences'].tolist()
labels = data_file['Label'].tolist()

snts, labels = word2vecVN(snts, labels)
SEQ_LEN = snts.size()[1]
BATCH_SIZE = 4

dataset = create_dataset(snts, labels)
loader = create_loader(dataset, BATCH_SIZE, True)

output_size = n_entity = len(token)
embedding_dim = 400
hidden_dim = 800
n_layers = 2
model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
data, label = next(iter(loader))


def aggregatenum(tensor):
	for i in range(len(tensor)):
		for j in range(len(tensor[0])):
			if tensor[i][j] == 7:
				tensor[i][j] = 2
			elif tensor[i][j] == 0:
				tensor[i][j] = 0
			else:
				tensor[i][j] = 1
	return tensor

pred = model(data).argmax(dim=2)
print(aggregatenum(pred))
print(aggregatenum(label))
print((pred==label).sum())
