import codecs
file  = codecs.open('hundred.csv','w',encoding='UTF-8')

str_in = [dg1+' trăm linh ' + dg2 + '\n' for dg1 in pre_num for dg2 in post_num]

for s in str_in:
	file.write(s)