from token import pre_num,flt
s = 'Tôi ăn cơm lúc mười giờ rưỡi'
def convert_to_num(str_in):
	words = str_in.split()
	cvt = {**pre_num,**flt}
	pre_ind = -3
	for i,w in enumerate(words):
		if w in cvt:
			words[i] = cvt[w]
			if pre_ind == i-2:
				words[pre_ind] += words[i]
				del words[i]
			pre_ind = i
	return ' '.join(words)
print(convert_to_num(s))
