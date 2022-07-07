import numpy as np
import pandas as pd
import torch
from data import *
from model import *
from utils import *

data_file = pd.read_csv('dataset.csv')
snts = data_file['Sentences'].tolist()
labels = data_file['Label'].tolist()
snts, labels = word2vecVN(snts, labels)
# SEQ_LEN = snts.size()[1]
# BATCH_SIZE = 4

# dataset = create_dataset(snts, labels)
# loader = create_loader(dataset, BATCH_SIZE, True)

# output_size = n_entity = 3
# embedding_dim = 400
# hidden_dim = 400
# n_layers = 1
# model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN)
# model.train(loader,4,BATCH_SIZE,0.005,'CE','Adam',2)
