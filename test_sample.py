#!/usr/bin/env python3

import time
import copy
import torch
import argparse
import warnings
import torch.nn as nn
import torch.optim as optim

from models import GCN, GAT, APPNP
from datasets import citation, Citation


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(model, optimizer, criterion, features, labels, adj, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = criterion(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return loss_train, acc_train, loss_val, acc_val, time.time() - t


def model_test(model, criterion, features, labels, adj, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test, acc_test


def test():    
    warnings.filterwarnings("ignore")
    class Args:
        def __init__(self):
            self.device = 'cpu'
            self.model = 'appnp'
            self.data_root = '.'
            self.dataset = 'cora'
            self.seed = 0
            self.epochs = 100
            self.lr = 0.01
            self.weight_decay = 5e-4
            self.hidden = 64
            self.dropout = 0.5

    args = Args()

    models = {'gcn':GCN, 'gat':GAT, 'appnp':APPNP}
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data = citation(root=args.data_root, name=args.dataset, device=args.device)
    adj, features, labels, idx_train, idx_val, idx_test, feat_len, num_class = data
    Model = models[args.model.lower()]
    model = Model(nfeat=feat_len, nhid=args.hidden, nclass=num_class, dropout=args.dropout).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    t_total = time.time()
    for epoch in range(args.epochs):
        loss_train, acc_train, loss_val, acc_val, t = train(model, optimizer, criterion, features, labels, adj, idx_train, idx_val)
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(t))

    print("Total time elapsed: {:.4f}s".format(time.time()-t_total))
    loss_test, acc_test = model_test(model, criterion, features, labels, adj, idx_test)
    print("Test results:", "loss: {:.4f}".format(loss_test.item()),
        "accuracy: {:.4f}".format(acc_test.item()))
    
    assert acc_test.item() > 0.83
