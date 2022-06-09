from utils import *

# s = 'Tôi có hai ba bốn cái kẹo'
# print(clean_num(s))

# file = open('testcase.csv', 'r', encoding='UTF-8')
# sntns = file.readlines()
# for sntn in sntns:
# 	print(sntn)
# 	print(convert_to_litnum(sntn))
# 	print('---------------------------------')
# file.close()

import pandas as pd
file1 = pd.read_csv('48_nghia_mic.csv')
file2 = pd.read_csv('2_hongnhu_mic.csv')
stns = pd.concat([file1['Groundtruth'],file1['Groundtruth']],axis=0)
for stn in stns:
	print(stn)
	print(token_label(stn.split()))
	print('---------------------------------')
# pd.concat([stns,stns.map(lambda x:' '.join(token_label(x.split())))],axis=1).to_csv('dataset.csv',index=False)