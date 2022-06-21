token = ['num', 'post_num', 'flt', 'num_mag', 'sub_num_mag', 'unit', 'unknown']

token2idx = {tok:i for i, tok in enumerate(token)}
idx2token = {i:tok for i, tok in enumerate(token)}
token2vec = {}
for key in token2idx.keys():
	vec = [0 for i in range(len(token))]
	vec[token2idx[key]] = 1
	token2vec[key] = vec

num = {
		'không':'0',
		'một':'1',
		'hai':'2',
		'ba':'3',
		'bốn':'4',
		'năm':'5',
		'sáu':'6',
		'bảy':'7',
		'tám':'8',
		'chín':'9',
		'mười':'10',
		'mười một':'11',
		'mười hai':'12',
		'mười ba':'13',
		'mười bốn':'14',
		'mười lăm':'15',
		'mười sáu':'16',
		'mười bảy':'17',
		'mười tám':'18',
		'mười chín':'19'
		}

post_num = {
		'mốt':'1',
		'tư':'4',
		'lăm':'5'
		}

flt = {
		'rưỡi':'0.5'
		}

num_mag = {
			'mươi':'10',
			'trăm':'100',
			'ngàn':'1000',
			'triệu':'1000000'
			}

sub_num_mag = ['lẻ','linh']

unit = ['kí lô gam', 'kí', 'cây số',
		'giờ', 'mét', 'ki lô mét', 'tạ',
		'yến', 'tá', 'héc ta', 'phút']
