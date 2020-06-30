# Graph Neural Networks
Efficient PyTorch implementation for graph neural networks (GNN).

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

   * [GAT: Graph Attention Network, ICLR 2018](https://arxiv.org/pdf/1710.10903.pdf)

   * Training

         python3 train.py --model GAT --epochs 100 --lr 0.01 --dataset cora 
         python3 train.py --model GAT --epochs 100 --lr 0.01 --dataset pubmed
         python3 train.py --model GAT --epochs 100 --lr 0.01 --dataset citeseer

## SAGE

   * [GraphSAGE: Inductive Representation Learning on Large Graphs, NIPS 2017](https://arxiv.org/pdf/1706.02216.pdf)

   * Training

         python3 train_sage.py --lr 0.01 --dataset cora
         python3 train_sage.py --lr 0.01 --dataset pubmed
         python3 train_sage.py --lr 0.01 --dataset citeseer

# Usage

* train.py

      usage: train.py [-h] [--device DEVICE] [--model MODEL] [--data-root DATA_ROOT]
                      [--dataset DATASET] [--seed SEED] [--epochs EPOCHS] [--lr LR]
                      [--weight_decay WEIGHT_DECAY] [--hidden HIDDEN]
                      [--dropout DROPOUT]

      optional arguments:
        -h, --help            show this help message and exit
        --device DEVICE       cpu, cuda:0, cuda:1, etc.
        --model MODEL         GCN/gcn or GAT/gat
        --data-root DATA_ROOT
                              dataset location
        --dataset DATASET     cora, citeseer, or pubmed
        --seed SEED           Random seed.
        --epochs EPOCHS       Number of epochs to train.
        --lr LR               Initial learning rate.
        --weight_decay WEIGHT_DECAY
                              Weight decay (L2 loss on parameters).
        --hidden HIDDEN       Number of hidden units.
        --dropout DROPOUT     Dropout rate.
      
* train_sage.py

      usage: train_sage.py [-h] [--device DEVICE] [--data-root DATA_ROOT]
                           [--dataset DATASET] [--save SAVE] [--aggr AGGR] [--lr LR]
                           [--factor FACTOR] [--min-lr MIN_LR] [--patience PATIENCE]
                           [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                           [--early-stop EARLY_STOP] [--weight_decay WEIGHT_DECAY]
                           [--seed SEED]

      optional arguments:
        -h, --help            show this help message and exit
        --device DEVICE       cpu, cuda:0, cuda:1, etc.
        --data-root DATA_ROOT
                              learning rate
        --dataset DATASET     cora, citeseer, pubmed
        --save SAVE           model file to save
        --aggr AGGR           Aggregator: mean, pool, or gcn
        --lr LR               learning rate
        --factor FACTOR       EarlyStopScheduler factor
        --min-lr MIN_LR       minimum lr for EarlyStopScheduler
        --patience PATIENCE   patience for Early Stop
        --batch-size BATCH_SIZE
                              number of minibatch size
        --epochs EPOCHS       number of training epochs
        --early-stop EARLY_STOP
                              number of epochs for early stop training
        --weight_decay WEIGHT_DECAY
                              Weight decay (L2 loss on parameters).
        --seed SEED           Random seed.
