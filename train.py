import numpy as np
import pandas as pd
import torch
from data import *
from model import *
from utils import *

# TRAIN DATA
data_file = pd.read_csv('num_refined_dataset.csv')
snts = data_file['Sentences'].tolist()
labels = data_file['Label'].tolist()
(snts,pos_tag), labels = word2vecVN(snts, labels)

BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
SEQ_LEN = snts.size()[1]
print(SEQ_LEN)

dataset = create_dataset(snts,pos_tag,labels)
loader = create_loader(dataset, BATCH_SIZE, True)


# TEST DATA
data_file = pd.read_csv('test_set.csv')
test_snts = data_file['Sentences'].tolist()
test_labels = data_file['Label'].tolist()
(test_snts,test_pos_tag), test_labels = word2vecVN(test_snts, test_labels)

test_dataset = create_dataset(test_snts,test_pos_tag,test_labels)
test_loader = create_loader(test_dataset, TEST_BATCH_SIZE, True)


# output_size = n_entity = len(token)
output_size = num_classes = 3
embedding_dim = 400
hidden_dim = 100
pos_dim = 100
n_layers = 1
# model = NER_LSTMNet(num_classes, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN, drop_prob=0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model = NER_BiLSTMNet(num_classes, embedding_dim, hidden_dim, pos_dim, seq_len=SEQ_LEN).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# model.load_state_dict(torch.load('./model.pt'))
model.train(loader,test_loader,epochs=500,batch_size=BATCH_SIZE,learning_rate=0.0025,criterion='CE',optimizer='Adam',clip=0.005)
