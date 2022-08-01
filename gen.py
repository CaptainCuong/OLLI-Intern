import pandas as pd

num = ['không','một','hai','ba','bốn','năm','sáu','bảy','tám','chín']
person = ['ông','bà','cô','dì','chú','bác','cậu','mợ','thím','anh','chị']
date1 = ['hai','ba','bốn','năm','sáu','bảy','tám','chín','mười']
date2 = ['mười một','mười hai','mười ba','mười bốn','mười năm','mười sáu','mười bảy','mười tám','mười chín',
		'hai mốt','hai hai','hai ba','hai bốn','hai năm','hai sáu','hai bảy','hai tám','hai chín','ba mươi','ba mốt']
post_num = ['mốt','hai','ba','bốn','tư','lăm','sáu','bảy','tám','chín']
strings = []
labels = []
for i in date1:
	strings.append('hát tê vê '+i)
	labels.append('unknown unknown unknown num')


file = pd.DataFrame({'Strings': strings,
					'Labels': labels})

file.to_csv('gen_data.csv',index=False)