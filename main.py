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
label_model_name = 'bilstm'
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)
SEQ_LEN = 20
output_size = n_entity = 4
embedding_dim = 400
hidden_dim = 400
n_layers = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if label_model_name == 'seq2seq':
    label_model = NER_LSTMNet(n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len=SEQ_LEN).to(device)
    label_model.load_state_dict(torch.load('model.pt', map_location=device))
elif label_model_name == 'bilstm':
    label_model = NER_BiLSTMNet(n_entity, embedding_dim, hidden_dim, seq_len = SEQ_LEN).to(device)
    label_model.load_state_dict(torch.load('bilstm_model.pt', map_location=device))
else:
    raise Exception('Invalid model for labelling')

abb_strings = [
            ('đi ăn ca ép xê','unknown unknown abb abb abb'),
            ('tập đoàn vi en pi ti','unknown unknown abb abb abb abb'),
            ('sơn tùng em ti pi','unknown unknown abb abb abb'),
            ('ét ti sơn thạch','abb abb unknown unknown'),
            ('ba công ty ép pi ti','num unknown unknown abb abb abb'),
            ('nộp xi vi cho công ti','unknown abb abb unknown unknown unknown'),
            ('ép pi ti di chuyển sang địa điểm mới','abb abb abb unknown unknown unknown unknown unknown unknown'),
            ('trường đại học ép tê diu ở đâu','unknown unknown unknown abb abb abb unknown unknown'),
            ('trường đại học hát xê mờ u tê','unknown unknown unknown abb abb abb abb abb'),
            ('công ty vàng bạc ét xi chây','unknown unknown unknown unknown abb abb abb'),
            ('anh nhớ em tê tái','unknown unknown unknown unknown unknown'),
            ('tổ chức đớp liu hát ô','unknown unknown abb abb abb abb'),
            ('anh ấy bị xi ai ây bắt','unknown unknown unknown abb abb abb unknown'),
            ('báo bi bi xi','unknown abb abb abb'),
            ('đài xi en en','unknown abb abb abb'),
            ('kĩ thuật mạng vi pi en','unknown unknown unknown abb abb abb'),
            ('đớp liu hát ô','abb abb abb abb'),
            ('ngân hàng vê pê bê','unknown unknown abb abb abb'),
            ('ba ngân hàng trên đường nờ tờ mờ ca','num unknown unknown unknown unknown abb abb abb abb'),
            ('em xi em ti pi','abb abb abb abb abb'),
            ('anh em','unknown unknown'),
            ('công ty vi en pi ti','unknown unknown abb abb abb abb'),
            ('mở đài hát tê vê ba lúc ba giờ chiều','unknown unknown abb abb abb num unknown num unknown unknown'),
            ]
num_strings = [
            # ('ba triệu năm trăm sáu mươi nghìn đồng','num num num num num num num unknown'),
            # ('triệu chứng bệnh lao phổi','unknown unknown unknown unknown unknown'),
            # ('hai mươi ba tháng bảy hai không hai mốt là thứ mấy','num num num unknown num num num num num unknown unknown unknown'),
            # ('mười bình cộng sáu mươi hai bằng bao nhiêu','num unknown unknown num num num unknown unknown unknown'),
            # ('tháng này có hai mươi ba ngày phải không','unknown unknown unknown num num num unknown unknown unknown'),
            # ('hai mươi chín năm sau','num num num unknown unknown'),
            # ('ba mươi hai năm sau','num num num unknown unknown'),
            # ('sau vài năm','unknown unknown unknown'),
            # ('năm hai ngàn không trăm linh bảy','unknown num num num num num num'),
            # ('tết nguyên đán hai không hai bốn ngày dương bao nhiêu','unknown unknown unknown num num num num unknown unknown unknown unknown'),
            # ('nhắc tôi về nhà trước chín giờ','unknown unknown unknown unknown unknown num unknown'),
            # ('nhắc tôi về nhà trước hai mốt giờ','unknown unknown unknown unknown unknown num num unknown'),
            # ('nhắc tôi đi ngủ trưa lúc hai giờ chiều','unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('hai triệu triệu chứng của bệnh nan y','num num unknown unknown unknown unknown unknown unknown'),
            # ('chín triệu không trăm ba mươi nghìn','num num num num num num num'),
            # ('năm triệu chín trăm hai mươi nghìn','num num num num num num num'),
            # ('ba đi làm lúc ba giờ chiều','unknown unknown unknown unknown num unknown unknown'),
            # ('ba em bắt con ba ba lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('tôi ăn chín quả trứng chín lúc chín giờ sáng','unknown unknown num unknown unknown unknown unknown num unknown unknown'),
            # ('anh hai ăn trưa lúc hai giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('triệu chứng hậu cô vít','unknown unknown unknown unknown unknown'),
            # ('một triệu năm','num num unknown'),
            # ('dì ba đi chợ lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown num unknown unknown'),
            # ('ba tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('mẹ tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('ba tôi đi làm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            # ('mở bài hát năm anh em trên một chiếc xe tăng','unknown unknown unknown num unknown unknown unknown unknown unknown unknown unknown'),
            # ('mở bài hát mười năm tình cũ','unknown unknown unknown num unknown unknown unknown'),
            # ('mở bài hát ba là cây nến vàng','unknown unknown unknown unknown unknown unknown unknown unknown'),
            # ('bật bài hà nội mười chín mùa hoa','unknown unknown unknown unknown unknown unknown unknown unknown'),
            # ('bật bài tuổi mười ba','unknown unknown unknown unknown unknown'),
            # ('mười chín tuổi rất chín chắn','num num unknown unknown unknown unknown'),
            # ('cuộc thi tổ chức năm hai ngàn không trăm lẻ chín','unknown unknown unknown unknown unknown num num num num num num'),
            # ('tắt quạt sau hai giờ','unknown unknown unknown num unknown'),
            # ('năm hai ngàn không trăm hai mươi bảy','unknown num num num num num num num'),
            # ('năm trăm mười bảy năm sau','num num num num unknown unknown'),
            # ('cô ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('dì ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('chú ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('bác ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('ông ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('bà ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('cậu ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('mợ ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('thím ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('anh ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('chị ba đang làm việc nhà lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            # ('một cây làm chẳng nên non ba cây chụm lại nên hòn núi cao','unknown unknown unknown unknown unknown unknown unknown unknown unknown unknown unknown unknown unknown unknown'),
            # ('anh hai mua hai mươi ba cây kẹo','unknown unknown unknown num num num unknown unknown'),
            # ('ngày thứ tư năm hai ngàn','unknown unknown unknown unknown num num'),
            # ('thứ sáu ngày mười bảy','unknown unknown unknown num num'),
            # ('ba em bắt con ba ba lúc ba giờ','unknown unknown unknown unknown unknown unknown unknown num unknown'),
            # ('hai chiếc xe trên đường mới đi ngang qua','num unknown unknown unknown unknown unknown unknown unknown unknown'),
            # ('năm anh em cùng nhau đi học','num unknown unknown unknown unknown unknown unknown'),
            # ('lặp lại bài hát năm anh em','unknown unknown unknown unknown unknown unknown unknown'),
            # ('phát lại bài hát ba chú cún con','unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('cho biết tám số lớn nhất có ba chữ số','unknown unknown num unknown unknown unknown unknown num unknown unknown'),
            ('bảy anh trai đang đi học','num unknown unknown unknown unknown unknown'),
            ]

true_lb = []
for num_str in num_strings:
    if not len(num_str) == 2:
        print(num_str)
    _, lbs = num_str
    for tag in lbs.split():
        true_lb.append(tag)
pred_lb = []

for string, true_label in num_strings:
    print('Original string:',string)
    converted, label = clean_abb(string, embedding_model = word2vec_model, label_model = label_model, seq_len = SEQ_LEN)
    pred_lb.extend(label.split())
    assert len(true_label.split()) == len(label.split()), ('True label len: %d, Predicted label len: %d')%(len(true_label.split()), len(label.split()))
    print('Label:\n',label,'\n')
    print('Converted string:',converted,'\n','-'*50,'\n')
    print('Accuracy :%f'%((np.sum(np.array(true_label.split()) == label.split()))/len(true_label.split())))

print(precision_recall_fscore_support(true_lb, pred_lb, labels=['abb','num','unknown'],average=None, zero_division=0))


ConfusionMatrixDisplay.from_predictions(true_lb, pred_lb)
plt.show()