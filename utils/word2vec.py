import re
from collections import Counter

import gensim
import nltk
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence

from .token import idx2token, token, token2idx, token2vec, spoken_alpb


def word2vec(sentences, labels):
	'''
	sentences: list(string)
	labels   : list(string)
	'''
	# Count frequency of each word
	cnt = Counter()
	for snt in sentences:
	    for wd in nltk.word_tokenize(snt):
	        cnt.update([wd])

	# Create a map from word to frequency and reverse
	word2idx = {k:v for k,v in cnt.items()}
	idx2word = {v:k for k,v in cnt.items()}
	print('MAP FROM WORD TO FREQUENCY:')
	print(word2idx)
	print('\n','-'*50,'\n')
	print(idx2word)


	for i, snt in enumerate(sentences):
	    sentences[i] = [word2idx[w] for w in nltk.word_tokenize(snt)]
	labels = [[token2idx[lb] for lb in nltk.word_tokenize(lbs)] for lbs in labels]
	return sentences, labels, len(cnt.items())

def tkvec2lbsnt(lb_sentences):
	'''
	[[1,0,2,3],
	 [5,6,1,0]]

	----->

	[[]]
	'''
	a = torch.argmax(lb_sentences, dim = 2)
	return [' '.join([idx2token[lb.item()] for lb in snt]) for snt in a]

def word2vecVN(sentences,labels_):
	model = 'baomoi.model.bin'
	word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)
	snts = []
	labels = []
	for snt, lbs in zip(sentences, labels_):
		print(snt, ':',lbs)
		snt = snt.lower().split()
		lbs = lbs.split()
		assert len(snt) == len(lbs), 'Sentence and label are not aligned\n' + ' '.join(snt)
		ws = []	
		for word in snt:
			try:
				ws.append(torch.tensor(word2vec_model.get_vector(word).reshape((1,-1))))
			except:
				if model == 'wiki.vi.model.bin':
					ws.append(torch.ones((400,0)))
				elif model == 'baomoi.model.bin':
					ws.append(torch.ones((400,0)))
		ws = torch.concat(ws, dim = 0)
		snts.append(ws)
		labelid = []
		for lb in lbs:
			if lb not in token:
				raise Exception('Label has not been defined')
			labelid.append(token2idx[lb])
		labels.append(torch.tensor(labelid))
	print(len(snts))
	print(len(labels))
	snts = pad_sequence(snts,batch_first=True)
	labels = pad_sequence(labels,batch_first=True) 
	def aggregatenum(tensor):
		for i in range(len(tensor)):
			for j in range(len(tensor[0])):
				if tensor[i][j] == 8:
					tensor[i][j] = 3
				elif tensor[i][j] == 7:
					tensor[i][j] = 2
				elif tensor[i][j] == 0:
					tensor[i][j] = 0
				else:
					tensor[i][j] = 1
		return tensor
	aggregatenum(labels)
	return snts, labels
