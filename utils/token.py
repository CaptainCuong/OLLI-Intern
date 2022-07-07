token = ('padding', 'num', 'post_num', 'flt', 'num_mag', 'sub_num_mag', 'unit', 'unknown','abb')
token2idx = {tok:i for i, tok in enumerate(token)}
idx2token = {i:tok for i, tok in enumerate(token)}
token2vec = {}
for key in token2idx.keys():
	vec = [0 for i in range(len(token))]
	vec[token2idx[key]] = 1
	token2vec[key] = vec

hard = ('không','triệu','năm','ba','hai','bốn','sáu','chín')

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
			'nghìn':'1000',
			'triệu':'1000000'
			}

num_mag_level = {
			'mươi':1,
			'trăm':2,
			'ngàn':3,
			'nghìn':3,
			'triệu':5
			}

sub_num_mag = ('lẻ','linh')

unit = ('kí lô gam', 'kí', 'cây số',
		'giờ', 'mét', 'ki lô mét', 'tạ',
		'yến', 'tá', 'héc ta', 'phút')

spoken_alpb = {
	'ây':'a',
	'bê':'b',
	'bi':'b',
	'xê':'c',
	'xi':'c',
	'đê':'d',
	'đi':'d',
	'e':'e',
	'i':'e',
	'ép':'f',
	'gờ':'g',
	'hát':'h',
	'hắc':'h',
	'chây':'j',
	'ca':'k',
	'khây':'k',
	'eo':'l',
	'lờ':'l',
	'em':'m',
	'mờ':'m',
	'en':'n',
	'nờ':'n',
	'âu':'o',
	'ô':'o',
	'o':'o',
	'pi':'p',
	'pờ':'p',
	'khiu':'q',
	'qui':'q',
	'a':'r',
	'rờ':'r',
	'ét':'s',
	'sờ':'s',
	'ti':'t',
	'tờ':'t',
	'ty':'t',
	'diu':'u',
	'du':'u',
	'u':'z',
	'vê':'v',
	'vi':'v',
	'đớp bồ diu':'w',
	'đớp bồ du':'w',
	'đấp bồ diu':'w',
	'đấp bồ du':'w',
	'đớp liu':'w',
	'vê kép':'w',
	'ích':'x',
	'ít':'x',
	'xờ':'x',
	'quai':'y',
	'dét':'z',
	'di':'z'
}

abbreviation_list = {
	'a':('abc','acb','abb','anz','ai','afc','apec','adb','aws'),
	'b':('bidv'),
	'c':('cv','cocc','cia'),
	'd':('dfid'),
	'e':('epu'),
	'f':('ftu','fbi','fao','fbi','flc'),
	'g':('gpt'),
	'h':('hsbc','hcm','hiv','hcmus','hcmut','hust','hup','hanu','hpu','hau','hlu','humg','hneu','nuae','hupes','hunre','huc','hmu','huph','hcma','hmu','hubt','hmtu'),
	'i':('imf','iaea','icc','iom','iunc'),
	'j':(''),
	'k':('kfc','kpmg'),
	'l':('lol','lgpt'),
	'm':('mhb','mc','mb'),
	'n':('nlp','nyu','ntn','nus','ntu'),
	'o':('ocb','omg','oecd','opec'),
	'p':(''),
	'q':('qc'),
	'r':(''),
	's':('st','scb','shb','ssi'),
	't':('tma','tm','tmu'),
	'u':('utc','usth','uefa','un','unesco','unfpa','unhcr','unicef','unido'),
	'v':('voh','vov','vng','vpb','vib','vbsp','vdb','vl','vnpt','vnu','vju','vj','viu','vnuuet','vnuued','vnuhus','vnuussh','vnuf','vnam','vnua','vwa'),
	'w':('who','wb','wfp','wto'),
	'x':(''),
	'y':(''),
	'z':('')
}

pronoun = ('ông','bà','cô','dì','chú','bác','cậu','mợ','thím','anh','chị')

abb_prior = ('nhóm', 'công ti', 'công ty', 'trường', 'học', 
			'doanh nghiệp', 'đoàn', 'hội', 'chương trình')