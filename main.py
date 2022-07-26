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
            ('dừng phát bài hát mười một phố phường','unknown unknown unknown unknown num num unknown unknown'),
            ('ba triệu năm trăm sáu mươi nghìn đồng','num num num num num num num unknown'),
            ('triệu chứng bệnh lao phổi','unknown unknown unknown unknown unknown'),
            ('hai mươi ba tháng bảy hai không hai mốt là thứ mấy','num num num unknown num num num num num unknown unknown unknown'),
            ('mười bình cộng sáu mươi hai bằng bao nhiêu','num unknown unknown num num num unknown unknown unknown'),
            ('tháng này có hai mươi ba ngày phải không','unknown unknown unknown num num num unknown unknown unknown'),
            ('hai mươi chín năm sau','num num num unknown unknown'),
            ('mua căn nhà ở quận hai','unknown unknown unknown unknown unknown num'),
            ('xe mang biển số năm sáu tám','unknown unknown unknown unknown num num num'),
            ('năm hai ngàn không trăm linh bảy','unknown num num num num num num'),
            ('tết nguyên đán hai không hai bốn ngày dương bao nhiêu','unknown unknown unknown num num num num unknown unknown unknown unknown'),
            ('nhắc tôi về nhà trước chín giờ','unknown unknown unknown unknown unknown num unknown'),
            ('nhắc tôi đi ngủ trước hai mốt giờ','unknown unknown unknown unknown unknown num num unknown'),
            ('nhắc tôi ăn cơm lúc hai giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            ('hai triệu triệu chứng của bệnh nan y','num num unknown unknown unknown unknown unknown unknown'),
            ('chín triệu không trăm ba mươi nghìn','num num num num num num num'),
            ('năm triệu chín trăm hai mươi nghìn','num num num num num num num'),
            ('ba đi làm lúc ba giờ chiều','unknown unknown unknown unknown num unknown unknown'),
            ('ba em bắt con ba ba lúc ba giờ chiều','unknown unknown unknown unknown unknown unknown unknown num unknown unknown'),
            ('tôi ăn chín quả trứng chín lúc chín giờ sáng','unknown unknown num unknown unknown unknown unknown num unknown unknown'),
            ('anh hai ăn trưa lúc hai giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            ('triệu chứng hậu cô vít','unknown unknown unknown unknown unknown'),
            ('giá cổ phiếu tăng hai mươi phần trăm','unknown unknown unknown unknown num num unknown unknown'),
            ('dì ba đi chợ lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            ('tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown num unknown unknown'),
            ('ba tôi ăn cơm lúc ba giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            ('mẹ tôi giặt đồ lúc năm giờ chiều','unknown unknown unknown unknown unknown num unknown unknown'),
            ('ba tôi lau nhà lúc sáu giờ tối','unknown unknown unknown unknown unknown num unknown unknown'),
            ('mở bài hát năm anh em trên một chiếc xe tăng','unknown unknown unknown num unknown unknown unknown unknown unknown unknown unknown'),
            ('dừng bài hát mười năm tình cũ','unknown unknown unknown num unknown unknown unknown'),
            ('phát bài hát ba là cây nến vàng','unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('bật bài hà nội mười chín mùa hoa','unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('mười chín tuổi rất chín chắn','num num unknown unknown unknown unknown'),
            ('cuộc thi tổ chức năm hai ngàn không trăm lẻ chín','unknown unknown unknown unknown unknown num num num num num num'),
            ('tắt quạt sau hai giờ','unknown unknown unknown num unknown'),
            ('năm hai ngàn không trăm hai mươi bảy','unknown num num num num num num num'),
            ('năm trăm mười bảy năm sau','num num num num unknown unknown'),
            ('cho biết tám số lớn nhất có ba chữ số','unknown unknown num unknown unknown unknown unknown num unknown unknown'),
            ('bảy anh trai đang đi học','num unknown unknown unknown unknown unknown'),
            ('tăng hai mức âm lượng','unknown num unknown unknown unknown'),
            ('tăng máy lạnh hai độ','unknown unknown unknown num unknown'),
            ('nhiệt độ ngoài trời có lớn hơn ba mươi mốt độ không','unknown unknown unknown unknown unknown unknown unknown num num num unknown unknown'),
            ('giảm máy lạnh hai độ','unknown unknown unknown num unknown'),
            ('tóp mười bài hát hay nhất tháng là bài gì','unknown num unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('thứ hai có lịch gì quan trọng không','unknown num unknown unknown unknown unknown unknown unknown'),
            ('lên lịch thứ ba có cuộc họp','unknown unknown unknown num unknown unknown unknown'),
            ('hai tin nổi bật nhất trong ngày','num unknown unknown unknown unknown unknown unknown'),
            ('thứ năm cần làm gì','unknown num unknown unknown unknown'),
            ('tháng bảy có công việc gì quan trọng không','unknown num unknown unknown unknown unknown unknown unknown unknown'),
            ('mười giờ cần làm gì','num unknown unknown unknown unknown'),
            ('mười chín tuổi có được lái xe không','num num unknown unknown unknown unknown unknown unknown'),
            ('ngày bốn tháng bảy hai năm trước có sự kiện gì','unknown num unknown num num unknown unknown unknown unknown unknown unknown'),
            ('mười một triệu mua được bộ bàn ghế nào','num num unknown unknown unknown unknown unknown unknown unknown'),
            ('số lớn nhất có bốn chữ số là số mấy','unknown unknown unknown unknown num unknown unknown unknown unknown unknown'),
            ('có hai mươi tư chữ cái trong tiếng việt','unknown num num num unknown unknown unknown unknown unknown'),
            ('năm ngư dân về bờ sau mười hai ngày trôi dạt','num unknown unknown unknown unknown unknown num num unknown unknown unknown'),
            ('số điểm mười giảm hơn bốn lần','unknown unknown num unknown unknown num unknown'),
            ('cầu thủ chơi ba mươi phút trong hiệp một','unknown unknown unknown num num unknown unknown unknown num'),
            ('ba chính sách mới về tiền lương từ tháng tám','num unknown unknown unknown unknown unknown unknown unknown unknown num'),
            ('trời nắng bốn mươi độ','unknown unknown num num unknown'),
            ('mặc ba lớp áo trong trời mùa đông','unknown num unknown unknown unknown unknown unknown unknown'),
            ('nữ sinh điểm mười môn ngữ văn viết mười một trang giấy','unknown unknown unknown num unknown unknown unknown unknown num num unknown unknown'),
            ('giải cứu chín người trong vụ hỏa hoạn','unknown unknown num unknown unknown unknown unknown unknown'),
            ('cuộc thi lần đầu tổ chức sau hai năm','unknown unknown unknown unknown unknown unknown unknown num unknown'),
            ('mười hai bình luận hay nhất của bài viết','num num unknown unknown unknown unknown unknown unknown unknown'),
            ('mở tập năm phim doremon','unknown unknown num unknown unknown'),
            ('chín mươi sáu triệu dân việt nam','num num num unknown unknown unknown unknown'),
            ('tphcm và hà nội có không quá năm phó chủ tịch ủy ban nhân dân','unknown unknown unknown unknown unknown unknown unknown num unknown unknown unknown unknown unknown unknown unknown'),
            ('số điểm mười giảm hơn bốn lần','unknown unknown num unknown unknown num unknown'),
            ('thành phố hồ chí minh cần một trăm năm mươi lao động','unknown unknown unknown unknown unknown unknown num num num num unknown unknown'),
            ('Bayern bỏ hai tám triệu Euro cho tiền đạo mười bảy tuổi','unknown unknown num num num unknown unknown unknown unknown num num unknown'),
            ('chín người mắc kẹt trong ngôi nhà bốc cháy','num unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('kỳ thi tốt nghiệp trung học phổ thông hai ngàn không trăm hai hai','unknown unknown unknown unknown unknown unknown unknown unknown num num num num num num'),
            ('ba điểm mười môn văn và nhiều điểm mười môn sử','num unknown num unknown unknown unknown unknown unknown num unknown unknown'),
            ('hai ngôi sao bóng đá của việt nam','num unknown unknown unknown unknown unknown unknown unknown'),
            ('chưa đầy hai mươi tư giờ sau thỏa thuận mới','unknown unknown num num num unknown unknown unknown unknown unknown'),
            ('hơn hai ngàn tám trăm thí sinh được điểm mười','unknown num num num num unknown unknown unknown unknown num'),
            ('tám cách đẩy lùi chứng mất ngủ','num unknown unknown unknown unknown unknown unknown'),
            ('giải cứu chín người trong đám cháy','unknown unknown num unknown unknown unknown unknown'),
            ('ba điểm đầu tay cho việt nam','num unknown unknown unknown unknown unknown unknown'),
            ('tóp mười nền văn minh cổ đại thần bí','unknown num unknown unknown unknown unknown unknown unknown unknown'),
            ('hơn năm mươi mốt phần trăm thí sinh đạt điểm dưới trung bình','unknown num num num unknown unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('tổ chức lại giao thông tại bốn nút giao trọng điểm','unknown unknown unknown unknown unknown unknown num unknown unknown unknown unknown'),
            ('tiếp tục phân luồng đến ngày hai mươi hai tháng mười','unknown unknown unknown unknown unknown unknown num num num unknown num'),
            ('Haaland mất mười hai phút để ghi bàn đầu tiên cho Man City','unknown unknown num num unknown unknown unknown unknown unknown unknown unknown unknown unknown'),
            ('Haaland ghi bàn nhiều gấp ba lần Ronaldo ở cùng độ tuổi','unknown unknown unknown unknown unknown num unknown unknown unknown unknown unknown unknown'),
            ('phổ điểm năm tổ hợp xét tuyển đại học','unknown unknown num unknown unknown unknown unknown unknown unknown'),
            ('trường tặng năm cuốn vở','unknown unknown num unknown unknown'),
            ('bông hoa điểm mười','unknown unknown unknown num'),
            ('mở bài hát cháu lên ba','unknown unknown unknown unknown unknown unknown'),
            ('phát bài hát cháu lên ba','unknown unknown unknown unknown unknown unknown'),
            ('phát bài hát cháu lên ba lúc ba giờ ngày hai ba','unknown unknown unknown unknown unknown unknown unknown num unknown unknown num num'),
            ('mở bài hát cháu lên ba ba lần ở phòng của ba','unknown unknown unknown unknown unknown unknown num unknown unknown unknown unknown unknown'),
            ('đọc bài thơ đồng tháp mười mười lần lúc mười giờ sáng','unknown unknown unknown unknown unknown unknown num unknown unknown num unknown unknown'),
            ('mở bài mười năm tình cũ hai mươi ba lần','unknown unknown unknown unknown unknown unknown num num num unknown'),
            ('nhắc tôi đi chơi vào thứ năm ngày năm tháng năm năm hai ngàn không trăm hai hai','unknown unknown unknown unknown unknown unknown num unknown num unknown num unknown num num num num num num'),
            ('ngày thứ ba thứ ba trong tháng ba là ngày mấy','unknown unknown  unknown unknown num unknown unknown num unknown unknown unknown'),
            ('ba người ba bắt ba con ba ba lúc ba giờ chiều ngày thứ ba','num unknown unknown unknown num unknown unknown unknown unknown num unknown unknown unknown unknown num'),
            ('mở bài hát hoa mười giờ','unknown unknown unknown unknown unknown unknown'),
            ('mở đài hát tê vê bảy','unknown unknown unknown unknown unknown num')
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
    print('Converted string:',converted,'\n')
    print('Accuracy :%f'%((np.sum(np.array(true_label.split()) == label.split()))/len(true_label.split())))
    print('-'*50,'\n')

print(precision_recall_fscore_support(true_lb, pred_lb, labels=['abb','num','unknown'],average=None, zero_division=0))
print('Overall Accuracy: %f'%(np.sum(np.array(true_lb) == np.array(pred_lb))/len(true_lb)))
print('Total correct lb: %f'%(np.sum(np.array(true_lb) == np.array(pred_lb))))
print('Total lb: %f'%(len(true_lb)))
# ConfusionMatrixDisplay.from_predictions(true_lb, pred_lb)
# plt.show()

TEST_BATCH_SIZE = 100
test_snts = [num_str for num_str, _ in num_strings]
test_labels = [label for _, label in num_strings]
(test_snts,test_pos_tag), test_labels = word2vecVN(test_snts, test_labels)
test_dataset = create_dataset(test_snts,test_pos_tag,test_labels)
test_loader = create_loader(test_dataset, TEST_BATCH_SIZE, True)


a = next(iter(test_loader))
if torch.cuda.is_available():
    a = a[0].cuda(), a[1].cuda(), a[2].cuda()
b = label_model(a[0], a[1]).argmax(dim=2)


valid_ele = 0
x = 0
print(a[0][0][:20])
for d1 in range(b.shape[0]):
  for d2 in range(b.shape[1]):
    if a[2][d1][d2] != 0:
      valid_ele += 1
      if a[2][d1][d2] == b[d1][d2]:
        x += 1
    else:
      break
# x = (a[2] == b).view(-1)
acc = x/valid_ele
print(acc)
print(x)
print(valid_ele)

# valid_ele = 0
# x = 0

# true_lb = []
# for num_str in num_strings:
#     if not len(num_str) == 2:
#         print(num_str)
#     _, lbs = num_str
#     true_lb.append([token2idx[tag] for tag in lbs.split()])

# for i in range(len(true_lb)):
#   for j in range(len(true_lb[i])):
#       valid_ele += 1
#       if a[2][i][j] == b[i][j]:
#         x += 1
# acc = x/valid_ele
# print(acc)
# print(x)
# print(valid_ele)
