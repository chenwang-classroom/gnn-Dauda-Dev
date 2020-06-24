#!/usr/bin/env python3

import torch
import torch.nn as nn


class SAGE(nn.Module):
    '''
    GraphSAGE: Inductive Representation Learning on Large Graphs, NIPS 2017
    https://arxiv.org/pdf/1706.02216.pdf
    '''
    def __init__(self, feat_len, num_class, hidden=128):
        super().__init__()
        self.tran1 = FeatTrans(feat_len, hidden)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.tran2 = FeatTrans(hidden, hidden)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(hidden, num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.tran1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.tran2(x, neighbor)
        return self.classifier(self.acvt2(x))


class FeatTrans(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.aggre = MeanAggregator()
        self.tranx = nn.Linear(in_features, out_features, False)
        self.trann = nn.Linear(in_features, out_features, False)

    def forward(self, x, neighbor):
        f = self.aggre(neighbor)
        x = self.tranx(x) + self.trann(f)
        neighbor = [self.tranx(n) for n in neighbor]
        return x, neighbor


class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, neighbor):
        return torch.cat([n.mean(dim=0, keepdim=True) for n in neighbor])
