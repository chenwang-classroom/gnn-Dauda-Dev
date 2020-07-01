#!/usr/bin/env python3

import torch.nn as nn


class APPNP(nn.Module):
    '''
    APPNP, ICLR 2019
    Predict then Propagate: Graph Neural Networks Meet Personalized Pagerank
    https://arxiv.org/pdf/1810.05997.pdf
    '''
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.1):
        super().__init__()
        self.tran = nn.Linear(nfeat, nhid)
        self.app1 = GraphAppnp(alpha)
        self.app2 = GraphAppnp(alpha)
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        h = self.tran(x)
        x = self.app1(h, adj, h)
        x = self.acvt(x)
        x = self.app2(x, adj, h)
        return self.classifier(x)


class GraphAppnp(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, adj, h):
        return (1 - self.alpha) * adj @ x + self.alpha * h
