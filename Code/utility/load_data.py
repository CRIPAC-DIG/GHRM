import numpy as np
import random as rd
from time import time
from math import log
import gc
import heapq
import copy
from utility.parser import parse_args
args = parse_args()

class Data(object):
    def __init__(self, path, batch_size):
        self.batch_size = batch_size

        doc_file = path + '/map.documents.txt'
        qrl_file = path + '/map.queries.txt'
        doc_dict_file = path + '/doc_dict.txt'
        qrl_dict_file = path + '/qrl_dict.txt'
        train_file = path + '/train_pairs/f{}.train.pairs'.format(str(args.fold))
        # test_file = path + '/valid_run/f{}.valid.run'.format(str(args.fold))
        test_file = path + '/test_run/f{}.test.run'.format(str(args.fold))
        match_file = path + '/qrels'
        unqiue_words = path + '/unique_words.txt'

        #get number of users and items
        self.n_docs, self.n_qrls, self.n_words = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.pos_pools = {}
        self.neg_pools = {}

        self.doc_dict = {}
        self.qrl_dict = {}
        self.doc_dict_rev = {}
        self.qrl_dict_rev = {}
        self.qrl_doc_match = {}

        self.doc_word_list = {}
        self.word_doc_list = {}
        self.qrl_word_list = {}
        self.doc_unqiue_word_list = {}

        self.word_freq = {}
        self.word_doc_freq = {}
        self.word_window_freq = {}
        self.doc_word_freq = {}
        self.qrl_word_freq = {}
        self.word_pair_count = {}
        self.num_window = 0
        self.all_neg = []

        window_size = 5
        self.windows = []

        print('loading documents...', end='', flush=True)
        with open(doc_file) as f, open(unqiue_words) as f2:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split('\t')
                    did = int(l[0])
                    words = [int(i) for i in l[1].split()]
                    self.doc_word_list[did] = words
                    self.n_words = max(self.n_words, max(words))
                    self.n_docs = max(self.n_docs, did)
            for l in f2.readlines():
                if len(l)>0:
                    l = l.strip().split('\t')
                    did = int(l[0])
                    words = [int(i) for i in l[1].split()]
                    self.doc_unqiue_word_list[did] = words

        print('done')
        
        print('loading queries...', end='', flush=True)
        with open(qrl_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split('\t')
                    qid = int(l[0])
                    words = [int(i) for i in l[1].split()]
                    self.qrl_word_list[qid] = words
                    self.n_words = max(self.n_words, max(words))
                    self.n_qrls = max(self.n_qrls, qid)

        print('done')

        self.n_docs += 1
        self.n_qrls += 1
        self.n_words += 1
        
        print('loading dict...', end='', flush=True)
        with open(doc_dict_file) as f1, open(qrl_dict_file) as f2:
            for l in f1.readlines():
                key, item = l.strip().split('\t')
                self.doc_dict[key] = int(item)
                self.doc_dict_rev[int(item)] = key
            for l in f2.readlines():
                key, item = l.strip().split('\t') 
                self.qrl_dict[key] = int(item)
                self.qrl_dict_rev[int(item)] = key

        with open(match_file) as f:
            for l in f.readlines():
                qrl_key, _, doc_key, score = l.strip().split()
                if qrl_key not in self.qrl_dict:
                    continue
                if doc_key not in self.doc_dict:
                    continue
                qrl = self.qrl_dict[qrl_key]
                doc = self.doc_dict[doc_key]
                if int(score) > 0: score = '1'
                if (qrl, score) in self.qrl_doc_match:
                    self.qrl_doc_match[(qrl, score)].append(doc)
                else:
                    self.qrl_doc_match[(qrl, score)] = [doc]
        print('done') 

        print('loading train&test set...', end='', flush=True)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip()
                    qrl_key, doc_key = l.split('\t')
                    qrl = self.qrl_dict[qrl_key]
                    doc = self.doc_dict[doc_key]
                    if qrl in self.train_items:
                        self.train_items[qrl].append(doc)
                    else:
                        self.train_items[qrl] = [doc]
                    self.n_train += 1

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip()
                    qrl_key, _, doc_key, _, _, _ = l.split()
                    qrl = self.qrl_dict[qrl_key]
                    doc = self.doc_dict[doc_key]
                    if qrl in self.test_set:
                        self.test_set[qrl].append(doc)
                    else:
                        self.test_set[qrl] = [doc]
                    self.n_test += 1
        print('done')

        for qrl in self.train_items.keys():
            for doc in self.train_items[qrl]:
                if not (qrl, '0') in self.qrl_doc_match:
                    self.qrl_doc_match[(qrl, '0')] = []
                if not (qrl, '1') in self.qrl_doc_match:
                    self.qrl_doc_match[(qrl, '1')] = []
                    self.all_neg.append(qrl)
                if doc in self.qrl_doc_match[(qrl, '0')] or doc in self.qrl_doc_match[(qrl, '1')]:
                    continue
                else:
                    self.qrl_doc_match[(qrl, '0')].append(doc)

        for q in self.all_neg:
            self.train_items.pop(q)
        self.positive_pool()
        self.negative_pool()
        print('init finish!')

    def positive_pool(self):
        t1 = time()
        for q in self.train_items.keys():
            self.pos_pools[q] = self.qrl_doc_match[(q, '1')]
        print('refresh positive pools', time() - t1)

    def negative_pool(self):
        t1 = time()
        for q in self.train_items.keys():
            self.neg_pools[q] = self.qrl_doc_match[(q, '0')]
        print('refresh negative pools', time() - t1)

    def sample(self):
        key_pool = list(self.train_items.keys())                                                                                                                                                                                                                                                                                                                                                                                                           
        rd.shuffle(key_pool)
        if self.batch_size <= len(key_pool):
            qrls = rd.sample(key_pool, self.batch_size)
        else:
            qrls = [rd.choice(key_pool) for _ in range(self.batch_size)]

        def sample_pos_docs_for_q_from_pools(q, num):
            pos_docs = self.pos_pools[q]
            return rd.sample(pos_docs, num)

        def sample_neg_docs_for_q_from_pools(q, num):
            neg_docs = self.neg_pools[q]
            return rd.sample(neg_docs, num)

        pos_docs, neg_docs = [], []
        for q in qrls:
            pos_docs += sample_pos_docs_for_q_from_pools(q, 1)
            neg_docs += sample_neg_docs_for_q_from_pools(q, 1)

        return qrls, pos_docs, neg_docs

    def print_statistics(self):
        print('n_docs=%d, n_qrls=%d, n_words=%d' % (self.n_docs, self.n_qrls, self.n_words))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_docs * self.n_qrls)))
