from .token import token, token2idx, idx2token, token2vec
from collections import Counter
import re
import nltk
import torch

def word2vec(sentences, labels):
	'''
	sentences: list(string)
	labels   : list(string)
	'''
	cnt = Counter()
	for snt in sentences:
	    for wd in nltk.word_tokenize(snt):
	        cnt.update([wd])

	word2idx = {k:v for k,v in cnt.items()}
	idx2word = {v:k for k,v in cnt.items()}

	for i, snt in enumerate(sentences):
	    sentences[i] = [word2idx[w] for w in nltk.word_tokenize(snt)]
	labels = [[token2vec[lb] for lb in nltk.word_tokenize(lbs)] for lbs in labels]
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
