#!/usr/bin/env python3

import math
import torch
import torch.nn as nn


class GCN(nn.Module):
    '''
    GCN: Graph Convolutional Network, ICLR 2017
    https://arxiv.org/pdf/1609.02907.pdf
    '''
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.acvt(x)
        x = self.gc2(x, adj)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, adj):
        return adj @ self.linear(input)
