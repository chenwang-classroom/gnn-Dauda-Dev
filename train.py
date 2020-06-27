#!/usr/bin/env python3

import time
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from models import GCN, GAT
from datasets import citation, Citation
from torch_util import EarlyStopScheduler


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(model, optimizer, criterion):
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


def test(model, criterion):
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test, acc_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0', help="cpu, cuda:0, cuda:1, etc.")
    parser.add_argument("--model", type=str, default='GCN', help="GCN or GAT")
    parser.add_argument("--data-root", type=str, default='.', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data = citation(root=args.data_root, name=args.dataset, device=args.device)
    adj, features, labels, idx_train, idx_val, idx_test, feat_len, num_class = data
    Model = GCN if args.model == 'GCN' else GAT
    model = Model(nfeat=feat_len, nhid=args.hidden, nclass=num_class, dropout=args.dropout).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    t_total = time.time()
    for epoch in range(args.epochs):
        loss_train, acc_train, loss_val, acc_val, t = train(model, optimizer, criterion)
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(t))

    print("Total time elapsed: {:.4f}s".format(time.time()-t_total))
    loss_test, acc_test = test(model, criterion)
    print("Test results:", "loss: {:.4f}".format(loss_test.item()),
        "accuracy: {:.4f}".format(acc_test.item()))
