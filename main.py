import gensim
from data import *
from gensim.models import KeyedVectors
from model import *
from utils import *

model = 'baomoi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)

output_size = n_entity = 3
embedding_dim = 400
hidden_dim = 400
n_layers = 1
label_model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=13)
label_model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

string = 'năm trăm năm mươi lăm'
print(clean_num(string, embedding_model = word2vec_model, label_model = label_model))
