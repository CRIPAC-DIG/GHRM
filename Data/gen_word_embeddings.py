import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse
import multiprocessing
import numpy as np
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
parser.add_argument('--dim', default=300, help='the dim of word embedding')
args = parser.parse_args()
path = './{}/'.format(args.dataset)
dim = args.dim

try:
    model = gensim.models.Word2Vec.load(path + 'word2vec_%d.model'%dim)
except:
    with open(path + 'clean.documents.txt') as f, open(path + 'clean.queries.txt') as f1, open(path + 'corpous.txt', 'w') as f2:
        lines = f.readlines()
        for line in lines:
            did, words = line.strip().split('\t')
            f2.writelines(words + '\n')
        lines = f1.readlines()
        for line in lines:
            qid, words = line.strip().split('\t')
            f2.writelines(words + '\n')

    model = Word2Vec(LineSentence(path + 'corpous.txt'), size=dim, window=5, min_count=1, workers=multiprocessing.cpu_count(),sg=1,iter=50)
    model.save(path + 'word2vec_%d.model'%dim)

word_dict = {}
with open(path+"word_dict.txt") as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip().split('\t')
            word = l[0]
            wid = int(l[1])
            word_dict[word] = wid

embeddings = []
for word in word_dict.keys():
    arr = np.array(model[word])
    embeddings.append(arr.reshape([1,-1]))
np.save(path+"word_embedding_%dd"%dim,np.concatenate(embeddings, axis=0))