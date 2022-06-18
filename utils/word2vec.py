from .token import token
from collections import Counter
import re
import nltk

def word2vec(sentences, labels):
	cnt = Counter()
	for snt in sentences:
	    for wd in nltk.word_tokenize(snt):
	        cnt.update([wd])

	word2idx = {k:v for k,v in cnt.items()}
	idx2word = {v:k for k,v in cnt.items()}
	token2ind = {tok:i+1 for i, tok in enumerate(token)}

	for i, snt in enumerate(sentences):
	    sentences[i] = [word2idx[w] for w in nltk.word_tokenize(snt)]
	labels = [[token2ind[lb] for lb in nltk.word_tokenize(lbs)] for lbs in labels]
	return sentences, labels, len(cnt.items())