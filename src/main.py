# encoding: utf-8

from utils import partition, SeqIndexer
import numpy as np
from keras.layers import embeddings
from keras.models import Sequential

text = """from their godly endeavour, assuring themselves,
that though bad Christians carp and repine
at the confidence and respect which Catholick
Princes and Ministers of State shew to their Clergy,
by trusting them with their conscienbces and affaires"""

sent_size = 6
matrix = []
indexer = SeqIndexer()
for sent in partition(text.split(), sent_size):
    idxs = indexer.encode(sent, size=sent_size)
    matrix.append(idxs)

matrix = np.array(matrix)
n, p = matrix.shape

# character indexer
indexer = SeqIndexer()
max_size = max([len(w) for w in text.split()])
for w in text.split():
    id_w = indexer.encode(w, size=max_size)
