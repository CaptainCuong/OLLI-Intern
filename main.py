import gensim
from data import *
from gensim.models import KeyedVectors
from model import *
from utils import *

model = 'baomoi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)

output_size = n_entity = 4
embedding_dim = 400
hidden_dim = 400
n_layers = 1
label_model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=13)
label_model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

# string1 = 'ba triệu năm trăm sáu mươi nghìn đồng'
# string2 = 'con ba ba'
# string3 = 'triệu chứng hậu cô vít'
# string4 = 'ba mươi tháng tư hai không hai hai là thứ mấy'
# string5 = 'triệu chứng ho lao'
# string6 = 'năm hai ngàn không trăm hai ba'
# string7 = 'tám bình cộng mười hai bằng bao nhiêu'
# string8 = 'tết nguyên đán hai không hai ba ngày dương bao nhiêu'
# string9 = 'số ba ba'
# string10 = 'ba ba'
# string11 = 'năm trăm năm sau'
# string12 = 'tháng này có ba mươi mốt ngày không'
# string13 = 'tháng này có ba mốt ngày không'
strings = [
            # 'ba triệu năm trăm sáu mươi nghìn đồng',
            # 'triệu chứng bệnh lao phổi',
            # 'hai mươi ba tháng bảy hai không hai mốt là thứ mấy',
            # 'mười bình cộng sáu mươi hai bằng bao nhiêu',
            # 'tháng này có hai mươi ba ngày phải không',
            # 'hai mươi chín năm sau',
            # 'ba mươi hai năm sau',
            # 'sau vài năm',
            # 'năm hai ngàn không trăm linh bảy',
            # 'tết nguyên đán hai không hai bốn ngày dương bao nhiêu',
            # 'nhắc tôi về nhà trước chín giờ',
            # 'nhắc tôi về nhà trước hai mốt giờ',
            # 'nhắc tôi đi ngủ trưa lúc hai giờ chiều',
            # 'hai triệu triệu chứng của bệnh nan y',
            # 'chín triệu không trăm ba mươi nghìn',
            # 'năm triệu chín trăm hai mươi nghìn',
            # 'có vui không',
            # 'ba đi làm lúc ba giờ chiều',
            # 'ba em bắt con ba ba lúc ba giờ chiều',
            # 'tui ăn chín quả trứng chín lúc chín giờ sáng',
            # 'anh hai ăn trưa lúc hai giờ chiều',
            'đi ăn ca ép xê',
            'tập đoàn vi en pi ti',
            'sơn tùng em ti pi',
            'ét ti sơn thạch',
            'ba công ty ép pi ti',
            'nộp xi vi cho công ti',
            'ép pi ti di chuyển sang địa điểm mới'
            ]


for string in strings:
    print('Original string:',string)
    converted, label = clean_num_abb(string, embedding_model = word2vec_model, label_model = label_model)
    print('Label:\n',label,'\n')
    print('Converted string:',converted,'\n','-'*50,'\n')


# str1 = 'hai trăm mười bốn'
# str2 = 'hai trăm mười lăm triệu'
# str3 = 'hai mươi ba nghìn bảy trăm linh hai'
# str4 = 'hai mươi ba nghìn bảy trăm ba mươi mốt'
# str5 = 'ba triệu năm trăm sáu mươi nghìn'
# print(lit2num(['mười','hai']))
# print(lit2num(str1.split()))
# print(lit2num(str2.split()))
# print(lit2num(str3.split()))
# print(lit2num(str4.split()))
# print(lit2num(str5.split()))
# print(lit2num('hai mươi chín'.split()))
