import os
import tqdm
import copy
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from models import SAGE
from datasets import Citation, graph_collate
from torch_util import count_parameters, EarlyStopScheduler


def performance(loader, net, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, neighbor) in enumerate(loader):
            if torch.cuda.is_available():
                inputs, targets, neighbor = inputs.to(device), targets.to(device), [item.to(device) for item in neighbor]
            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = correct/total
    return acc


def train(loader, net, criterion, optimizer, device):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets, neighbor) in enumerate(loader):
        inputs, targets, neighbor = inputs.to(device), targets.to(device), [item.to(device) for item in neighbor]
        optimizer.zero_grad()
        outputs = net(inputs, neighbor)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return (train_loss/(batch_idx+1), correct/total)


if __name__ == '__main__':
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="learning rate")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, pubmed")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--aggr", type=str, default='mean', help="Aggregator: mean, pool, or gcn")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="EarlyStopScheduler factor")
    parser.add_argument("--min-lr", type=float, default=0.001, help="minimum lr for EarlyStopScheduler")
    parser.add_argument("--patience", type=int, default=10, help="patience for Early Stop")
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5e-4, help="number of epochs for early stop training")
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    # Datasets
    train_data = Citation(root=args.data_root, name=args.dataset, data_type='train')
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate)
    val_data = Citation(root=args.data_root, name=args.dataset, data_type='val')
    val_loader = Data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)
    test_data = Citation(root=args.data_root, name=args.dataset, data_type='test')
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)

    # Models
    net = SAGE(feat_len=train_data.feat_len, num_class=train_data.num_class, aggr=args.aggr).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    # Training
    print('number of parameters:', count_parameters(net))
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, optimizer, args.device)
        val_acc = performance(val_loader, net, args.device) # validate
        print("epoch: %d, train_loss: %.4f, train_acc: %.3f, val_acc: %.3f"
                % (epoch, train_loss, train_acc, val_acc))

        if val_acc > best_acc:
            print("New best Model, copying...")
            best_acc, best_net = val_acc, copy.deepcopy(net)

        if scheduler.step(error=1-val_acc):
            print('Early Stopping!')
            break

    train_acc, test_acc = performance(train_loader, best_net, args.device), performance(test_loader, best_net, args.device)
    print('train_acc: %.3f, test_acc: %.3f'%(train_acc, test_acc))
