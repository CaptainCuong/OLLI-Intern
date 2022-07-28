import gensim
import matplotlib.pyplot as plt
import numpy as np
from data import *
from gensim.models import KeyedVectors
from model import *
from sklearn.metrics import (ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
from utils import *
from utils.token import token2idx
import  pandas as pd 
model = 'baomoi.model.bin'
label_model_name = 'bilstm'
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)
SEQ_LEN = 30
output_size = n_entity = 3
embedding_dim = 400
hidden_dim = 100
pos_dim = 100
n_layers = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if label_model_name == 'seq2seq':
    label_model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN).to(device)
    label_model.load_state_dict(torch.load('model.pt', map_location=device))
elif label_model_name == 'bilstm':
    label_model = NER_BiLSTMNet(n_entity, embedding_dim, hidden_dim, pos_dim, seq_len = SEQ_LEN).to(device)
    label_model.load_state_dict(torch.load('bilstm_model.pt', map_location=device))
else:
    raise Exception('Invalid model for labelling')


file = pd.read_csv('data.csv')
num_strings = file['Sentences']
converted_str = []
for string in num_strings:
    print('Original string:',string)
    converted, label = clean_abb(string, embedding_model = word2vec_model, label_model = label_model, seq_len = SEQ_LEN)
    converted_str.append(converted.upper())


out_file = pd.DataFrame({'sid':list(range(1,len(num_strings)+1)),
                         'orig_text':num_strings,
                         'post_text':converted_str})
out_file.to_csv('data_converted.csv',index=False)