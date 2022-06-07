from .token import num, post_num, flt, num_mag, sub_num_mag, unit

def convert_to_litnum(str_in):
	words = str_in.split()
	swap_flt(words)
	# print(words)
	words = lit2num(words)
	return ' '.join(words)

def token_label(ws):
	words = ws.copy()
	for i,w in enumerate(words):
		if w in num:
			words[i] = 'num'
		elif w in post_num:
			words[i] = 'post_num'
		elif w in flt:
			words[i] = 'flt'
		elif w in unit:
			words[i] = 'unit'
		elif w in num_mag:
			words[i] = 'num_mag'
		elif w in sub_num_mag:
			words[i] = 'sub_num_mag'
		else:
			words[i] = 'unknown'
	return words

def swap_flt(words):
	for i in swap_flt_ind(token_label(words)):
		temp = words[i]
		words[i] = words[i-1]
		words[i-1] = temp

def swap_flt_ind(words):
	for i in range(1, len(words)):
		if words[i] == 'flt':
			yield i

def clean_num(words):
	i = 0
	rt = []
	words = words.split()
	while i < len(words):
		if words[i] in num | post_num | flt | num_mag or words[i] in sub_num_mag:
			st = i
			while i < len(words) and (words[i] in num | post_num | flt | num_mag or words[i] in sub_num_mag):
				i += 1
			lt = i
			rt.append(lit2num(words[st:lt]))
			i -= 1
		else:
			rt.append(words[i])
		i += 1
	return ' '.join(rt)

def lit2num(words):
	i = 0
	stk = [0]
	# words = words.split()
	swap_flt(words)
	while i < len(words):
		if words[i] in num:
			num_con = []
			while i < len(words) and words[i] in num:
				num_con.append(num[words[i]])
				i += 1
			i -= 1
			stk.append(float(''.join(num_con)))

		elif words[i] in num_mag:
			stk[-1] = stk[-1]*float(num_mag[words[i]])

		elif words[i] in flt:
			stk[-1] += float(flt[words[i]])

		i += 1
	return str(sum(stk))
