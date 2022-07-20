import gensim
import matplotlib.pyplot as plt
import numpy as np
from data import *
from gensim.models import KeyedVectors
from model import *
from sklearn.metrics import (ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
from utils import *

model = 'baomoi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)

output_size = n_entity = 4
embedding_dim = 400
hidden_dim = 400
n_layers = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
label_model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=13).to(device)
label_model = NER_BiLSTMNet(n_entity, embedding_dim, hidden_dim, seq_len=13).to(device)
label_model.load_state_dict(torch.load('bilstm_model.pt', map_location=device))
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
            # ('ba mươi hai năm sau','num num num unknown unknown'),
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
            # ('ba đi làm lúc ba giờ chiều','unknown unknown unknown unknown num unknown unknown'),
            # ('ba em bắt con ba ba lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('tui ăn chín quả trứng chín lúc chín giờ sáng','unknown unknown num unknown unknown unknown unknown num unknown unknown'),
            # ('anh hai ăn trưa lúc hai giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('đi ăn ca ép xê','unknown unknown abb abb abb'),
            # ('tập đoàn vi en pi ti','unknown unknown abb abb abb abb'),
            # ('sơn tùng em ti pi','unknown unknown abb abb abb'),
            # ('ét ti sơn thạch','abb abb unknown unknown'),
            # ('ba công ty ép pi ti','num unknown unknown abb abb abb'),
            # ('nộp xi vi cho công ti','unknown abb abb unknown unknown unknown'),
            # ('ép pi ti di chuyển sang địa điểm mới','abb abb abb unknown unknown unknown unknown unknown unknown'),
            # ('trường đại học ép tê diu ở đâu','unknown unknown unknown abb abb abb unknown unknown'),
            # ('trường đại học hát xê mờ u tê','unknown unknown unknown abb abb abb abb abb'),
            # ('công ty vàng bạc ét xi chây','unknown unknown unknown unknown abb abb abb'),
            # ('anh nhớ em tê tái','unknown unknown unknown unknown unknown'),
            # ('tổ chức đớp liu hát ô','unknown unknown abb abb abb abb'),
            # ('anh ấy bị xi ai ây bắt','unknown unknown unknown abb abb abb unknown'),
            # ('báo bi bi xi','unknown abb abb abb'),
            # ('đài xi en en','unknown abb abb abb'),
            # ('kĩ thuật mạng vi pi en','unknown unknown unknown abb abb abb'),
            # ('đớp liu hát ô','abb abb abb abb'),
            # ('ngân hàng vê pê bê','unknown unknown abb abb abb'),
            # ('ba ngân hàng trên đường nờ tờ mờ ca','num unknown unknown unknown unknown abb abb abb abb'),
            # ('em xi em ti pi','abb abb abb abb abb'),
            # ('anh em','unknown unknown'),
            # ('công ty vi en pi ti','unknown unknown abb abb abb abb'),
            # ('mở đài hát tê vê ba lúc ba giờ chiều','unknown unknown abb abb abb num unknown num unknown unknown'),
            # ('tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown num unknown unknown'),
            # ('ba tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('mẹ tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('ba tôi đi làm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown')
            # ('mở bài hát năm anh em trên một chiếc xe tăng','unknown unknown unknown num unknown unknown unknown unknown unknown unknown unknown')
            ('mở bài hát mười năm tình cũ','unknown unknown unknown num unknown unknown unknown')

            # ('công ty vi en pi ti',''),
            # ('mờ hát',''),
            # ('mở đài vê ô hát',''),
            # ('mờ đài vê tê vê ba',''),
            ]

true_lb = [tag for _, lbs in strings for tag in lbs.split()]
pred_lb = []

for string, _ in strings:
    print('Original string:',string)
    converted, label = clean_abb(string, embedding_model = word2vec_model, label_model = label_model)
    pred_lb.extend(label.split())
    print('Label:\n',label,'\n')
    print('Converted string:',converted,'\n','-'*50,'\n')

print(precision_recall_fscore_support(true_lb, pred_lb, labels=['abb','num','unknown'],average=None, zero_division=0))


ConfusionMatrixDisplay.from_predictions(true_lb, pred_lb)
plt.show()


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
