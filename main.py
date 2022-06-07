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
file = pd.read_csv('48_nghia_mic.csv')
stns = file['Groundtruth']
for stn in stns:
	print(stn)
	print(clean_num(stn))
	print('---------------------------------')

