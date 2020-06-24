import math
import torch
import numpy as np
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1), nn.Dropout(dropout))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        h = self.tran(x)
        e = self.att1(h).unsqueeze(0) + self.att2(h).unsqueeze(1)
        e = self.leakyrelu(e.squeeze())
        e[adj.to_dense()<=0] = -math.inf # only neighbors
        return self.norm(e) @ h


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        self.atts = [Attention(nfeat, nhid, dropout, alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.atts):
            self.add_module('attention_{}'.format(i), attention)

        self.att = Attention(nhid * nheads, nclass, dropout, alpha)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.acvt = nn.ELU()

    def forward(self, x, adj):
        x = torch.cat([att(self.dropout1(x), adj) for att in self.atts], dim=1)
        return self.att(self.dropout2(self.acvt(x)), adj)
