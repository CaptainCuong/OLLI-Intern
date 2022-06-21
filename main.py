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
SEQ_LEN = 50
snts = nml_sentence(SEQ_LEN, snts)
labels = nml_label(SEQ_LEN, labels)

BATCH_SIZE = 4
dataset = create_dataset(torch.from_numpy(snts), torch.from_numpy(labels))
loader = create_loader(dataset, BATCH_SIZE, True)

output_size = n_entity = len(token)
embedding_dim = 128
hidden_dim = 256
n_layers = 2
model = NER_LSTMNet(vocab_size, n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN)
model.train(loader,10,BATCH_SIZE,0.005,'BCE','Adam',2)
torch.save(model.state_dict(), './model.pt')


