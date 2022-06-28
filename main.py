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

string1 = 'ba triệu năm trăm sáu mươi nghìn đồng'
string2 = 'con ba ba'
string3 = 'triệu chứng hậu cô vít'
string4 = 'ba mươi tháng tư hai không hai hai là thứ mấy'
string5 = 'triệu chứng ho lao'
string6 = 'năm hai ngàn không trăm hai ba'
string7 = 'tám bình cộng mười hai bằng bao nhiêu'
converted, label = clean_num(string7, embedding_model = word2vec_model, label_model = label_model)
print('Original string:\n',string7,'\n','-'*50)
print('Converted string:\n',converted,'\n','-'*50)
print('Label:\n',label)
