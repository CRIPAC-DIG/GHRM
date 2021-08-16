import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os
import warnings 
from tqdm import tqdm, trange
import multiprocessing
import argparse
warnings.filterwarnings("ignore") 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
parser.add_argument('--window_size', default=5, help='window size')
parser.add_argument('--doc_pad_len', default=300, help='document pad length')
args = parser.parse_args()
path = './{}/'.format(args.dataset)

doc_file = path + '/map.documents.txt'
qrl_file = path + '/map.queries.txt'
squeeze_words = path + '/doc_word_list_squeezed.txt'
n_docs, n_words = 0, 0
doc_word_list = []
qrl_word_list = []
doc_word_list_squeezed = []
DOC_PAD_LEN = args.doc_pad_len

window_size = args.window_size
windows = []

with open(doc_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip().split('\t')
            did = int(l[0])
            words = [int(i) for i in l[1].split()]
            length = len(words)
            doc_word_list.append((did, words))
            # slide windows
            window = []
            for j in range(length - window_size + 1):
                window += [words[j: j + window_size]]
            windows.append(window)
            n_words = max(n_words, max(words))
            n_docs = max(n_docs, did)
n_docs += 1
n_words += 1

def pad_sequences(items, maxlen, value=n_words):
    result = []
    for item in items:
        if len(item) < maxlen:
            item = item + [value] * (maxlen - len(item))
        if len(item) > maxlen:
            item = item[:maxlen]
        result.append(item)
    return result
def normalized_adj_bi(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

doc_word_list_squeezed = [sorted(set(doc_word_list[docid][1]), key=doc_word_list[docid][1].index) for docid in range(n_docs)]
with open(squeeze_words, 'w') as f:
    for i in range(len(doc_word_list_squeezed)):
        f.writelines(str(i)+'\t')
        for word in doc_word_list_squeezed[i]:
            f.writelines(str(word)+' ')
        f.writelines('\n')

padded_words = pad_sequences(doc_word_list_squeezed, DOC_PAD_LEN)
def func(start, end):
    if end > n_docs: end=n_docs
    batch_adj = []
    for k in trange(start, end):
        R = sp.dok_matrix((n_words+1, n_words+1), dtype=np.float32) 
        for window in windows[k]:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_j = window[j]
                    if word_i == word_j:
                        continue
                    R[word_i,word_j] += 1.
                    R[word_j,word_i] += 1.
        R = R.tocsc()
        adj = R[padded_words[k],:][:,padded_words[k]].toarray()
        batch_adj.append(normalized_adj_bi(adj))
    return np.stack(batch_adj, 0)

# # multiprocessing for large collections:
pool = multiprocessing.Pool(processes=20)
res = []
t = n_docs // 20
for i in range(21):
    res.append(pool.apply_async(func, (i*t,(i+1)*t)))
pool.close()
pool.join()

r = [i.get() for i in res]
arr = np.concatenate(r, 0)
np.save(path + "/doc_adj.npy",arr)
