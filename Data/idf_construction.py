import numpy as np
from math import log
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
args = parser.parse_args()
path = './{}/'.format(args.dataset)

doc_file = path + '/map.documents.txt'
qrl_file = path + '/map.queries.txt'
squeeze_words = path + '/doc_word_list_squeezed.txt'
n_docs, n_words = 0, 0
doc_word_list = []
word_doc_list = {}
vocab = set()
word_freq = {}
doc_word_freq = {}


with open(doc_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip().split('\t')
            did = int(l[0])
            words = [int(i) for i in l[1].split()]
            length = len(words)
            doc_word_list.append((did, words))
            for word in words:
                vocab.add(word)
                # total word frequency
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
                # in which document the word appears
                if word in word_doc_list:
                    word_doc_list[word].append(did)
                else:
                    word_doc_list[word] = [did]
                # word frequency in each document
                doc_word = (did, word)
                if doc_word in doc_word_freq:
                    doc_word_freq[doc_word] += 1
                else:
                    doc_word_freq[doc_word] = 1
            n_words = max(n_words, max(words))
            n_docs = max(n_docs, did)
n_docs += 1
n_words += 1

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(set(doc_list))

idf = {}
for word in range(n_words+1):
    try:
        idf[word] = log(1.0 * len(doc_word_list) / word_doc_freq[word])
    except:
        idf[word] = 0.

np.save(path + "idf.npy", idf)