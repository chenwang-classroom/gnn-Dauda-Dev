# GNN Tutorial
Tutorial for graph neural network

# Dependencies

    * PyTorch
    * DGL (Only used for automatically dataset downloading)

# Training

## GCN
    
    * [GCN: Graph Convolutional Network, ICLR 2017](https://arxiv.org/pdf/1609.02907.pdf)
    
    * Training

        python3 train.py --model GCN --epochs 200 --lr 0.01 --dataset cora 
        python3 train.py --model GCN --epochs 200 --lr 0.01 --dataset pubmed
        python3 train.py --model GCN --epochs 200 --lr 0.01 --dataset citeseer

## GAT

    * [Graph Attention Network, ICLR 2018](https://arxiv.org/pdf/1710.10903.pdf)

    * Training

        python3 train.py --model GAT --epochs 100 --lr 0.01 --dataset cora 
        python3 train.py --model GAT --epochs 100 --lr 0.01 --dataset pubmed
        python3 train.py --model GAT --epochs 100 --lr 0.01 --dataset citeseer

## SAGE

    * [GraphSAGE: Inductive Representation Learning on Large Graphs, NIPS 2017](https://arxiv.org/pdf/1706.02216.pdf)

    * Training

        python3 train_sage.py --epochs 100 --lr 0.01 --dataset cora 
        python3 train_sage.py --epochs 100 --lr 0.01 --dataset pubmed
        python3 train_sage.py --epochs 100 --lr 0.01 --dataset citeseer
