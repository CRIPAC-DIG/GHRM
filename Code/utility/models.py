import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utility.parser import parse_args
from utility.batch_test import data_generator, test
args = parse_args()

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_docs = config['n_docs']
        self.n_qrls = config['n_qrls']
        self.n_words = config['n_words']
        pretrained_embeddings = np.load("../Data/{}/word_embedding_300d.npy".format(args.dataset))
        l2_norm = np.sqrt((pretrained_embeddings * pretrained_embeddings).sum(axis=1))
        pretrained_embeddings = pretrained_embeddings / l2_norm[:, np.newaxis]
        pretrained_embeddings = np.concatenate([pretrained_embeddings, np.zeros([1, pretrained_embeddings.shape[-1]])], 0)
        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings),freeze=True,padding_idx=self.n_words).cuda()
        self.word_embedding = self.word_embedding.float()
        self._init_weights()
    def _init_weights(self):
        raise NotImplementedError 
    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)
    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
    def create_simmat(self, a_emb, b_emb):
        BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
        a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9 # avoid 0div
        b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9 # avoid 0div
        perm = b_emb.permute(0, 2, 1)
        sim = a_emb.bmm(perm)
        sim = sim / (a_denom * b_denom)
        return sim


class GHRM(BaseModel):
    def _init_weights(self):
        self.docs_adj = self.config['docs_adj']
        self.idf_dict = self.config['idf_dict']
        self.linear1 = nn.Linear(args.topk*3, 64).cuda()
        self.linear2 = nn.Linear(64, 32).cuda()
        self.linear3 = nn.Linear(32, 1).cuda()

        self.linearz0 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearz1 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr0 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr1 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh0 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh1 = nn.Linear(args.qrl_len, args.qrl_len).cuda()

        self.linearz02 = nn.Linear(1,1).cuda()
        self.linearz12 = nn.Linear(1,1).cuda()
        self.linearr02 = nn.Linear(1,1).cuda()
        self.linearr12 = nn.Linear(1,1).cuda()
        self.linearh02 = nn.Linear(1,1).cuda()
        self.linearh12 = nn.Linear(1,1).cuda()

        self.linearz03 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearz13 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr03 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr13 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh03 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh13 = nn.Linear(args.qrl_len, args.qrl_len).cuda()

        self.linearz04 = nn.Linear(1,1).cuda()
        self.linearz14 = nn.Linear(1,1).cuda()
        self.linearr04 = nn.Linear(1,1).cuda()
        self.linearr14 = nn.Linear(1,1).cuda()
        self.linearh04 = nn.Linear(1,1).cuda()
        self.linearh14 = nn.Linear(1,1).cuda()

        self.gated = nn.Linear(1, 1).cuda()
        self.linearp1 = nn.Linear(args.qrl_len,1).cuda()
        self.linearp2 = nn.Linear(args.qrl_len,1).cuda()
        #self.dropout = nn.Dropout(args.dp)
    def ggnn1(self,feat,doc_ids):
        adj = self.docs_adj[doc_ids]
        adj = torch.FloatTensor(adj).cuda()
        x = feat
        a = adj.matmul(x)

        z0 = self.linearz0(a)
        z1 = self.linearz1(x)
        z = F.sigmoid(z0 + z1)

        r0 = self.linearr0(a)
        r1 = self.linearr1(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh0(a)
        h1 = self.linearh1(r*x)
        h = F.relu(h0 + h1)

        feat = h*z + x*(1-z)
        #x = self.dropout(feat)
        return feat
    def ggnn2(self,feat,doc_ids):
        adj = self.docs_adj[doc_ids]
        adj = torch.FloatTensor(adj).cuda()
        x = self.linearp1(feat)
        a = adj.matmul(x)

        z0 = self.linearz02(a)
        z1 = self.linearz12(x)
        z = F.sigmoid(z0 + z1)
        
        r0 = self.linearr02(a)
        r1 = self.linearr12(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh02(a)
        h1 = self.linearh12(r*x)

        h = F.relu(h0 + h1)
        
        feat_s = h*z + x*(1-z)

        score, indices = feat_s.topk(240,1)
        indices = torch.squeeze(indices)
        seq_feat = []
        seq_adj = []
        for i in range(feat.shape[0]):
            seq_feat.append(feat[i][indices[i]])
            seq_adj.append(adj[i,indices[i],:][:,indices[i]])
        feat = torch.stack(tuple(seq_feat))
        adj = torch.stack(tuple(seq_adj))
        feat = F.tanh(score) * feat
        return feat, adj
    
    def ggnn3(self,feat,adj):
        x = feat
        a = adj.matmul(x)
        
        z0 = self.linearz03(a)
        z1 = self.linearz13(x)
        z = F.sigmoid(z0 + z1)

        r0 = self.linearr03(a)
        r1 = self.linearr13(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh03(a)
        h1 = self.linearh13(r*x)

        h = F.relu(h0 + h1)
        
        feat = h*z + x*(1-z)
        return feat

    def ggnn4(self,feat,adj):
        x = self.linearp2(feat)
        a = adj.matmul(x)
        
        z0 = self.linearz04(a)
        z1 = self.linearz14(x)
        z = F.sigmoid(z0 + z1)
        
        r0 = self.linearr04(a)
        r1 = self.linearr14(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh04(a)
        h1 = self.linearh14(r*x)

        h = F.relu(h0 + h1)
        
        feat_s = h*z + x*(1-z)
        score, indices = feat_s.topk(192,1)
        indices = torch.squeeze(indices)
        seq_feat = []
        #seq_adj = []
        for i in range(feat.shape[0]):
            seq_feat.append(feat[i][indices[i]])
            #seq_adj.append(adj[i,indices[i],:][:,indices[i]])
        feat = torch.stack(tuple(seq_feat))
        #adj = torch.stack(tuple(seq_adj))
        feat = F.tanh(score) * feat
        return feat#, adj
    
    def forward(self, qrl_token, doc_token, doc_ids, test=False):
        self.test = test
        self.batch_size = len(qrl_token)
        self.idf = torch.FloatTensor([[self.idf_dict[word] for word in words] for words in qrl_token]).cuda().unsqueeze(-1)
        qrl_word_embedding = self.word_embedding(torch.tensor(qrl_token).long().cuda())
        doc_word_embedding = self.word_embedding(torch.tensor(doc_token).long().cuda())
        feat = self.create_simmat(qrl_word_embedding, doc_word_embedding).permute(0, 2, 1) #batch, len_d, len_q
        
        feat_per = feat.permute(0,2,1)
        topk_0, _ = feat_per.topk(args.topk,-1)

        rep1 = self.ggnn1(feat, doc_ids)
        att_x1, adj_new = self.ggnn2(rep1, doc_ids)

        rep3 = self.ggnn3(att_x1,adj_new)
        att_x2 = self.ggnn4(rep3,adj_new)

        #1-hop representation
        att_x1 = att_x1.permute(0,2,1)  #batch, qrl, doc
        att_x1, _ = att_x1.topk(args.topk,-1)
        #2-hop representation
        att_x2 = att_x2.permute(0,2,1) 
        att_x2, _ = att_x2.topk(args.topk,-1) 

        att_x = torch.cat((topk_0,att_x1,att_x2),dim=-1)
        rel = F.relu(self.linear1(att_x))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        if args.idf:
            gated_weight = F.softmax(self.gated(self.idf), dim=1)
            rel = rel * gated_weight
        scores = rel.squeeze(-1).sum(-1, keepdim=True)
        if test:
            scores = scores.reshape((1, -1))
        return scores
