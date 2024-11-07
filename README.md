[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/h3ktCkJ5)
# Graph Neural Networks
Efficient PyTorch implementation for graph neural networks (GNN).

# Dependencies

   * [PyTorch](https://pytorch.org/get-started/locally)
   * [DGL==0.4.3](https://www.dgl.ai/pages/start.html) (Only used for automatically dataset downloading)
   
          pip3 install -r requirements.txt
   Note that only DGL==0.4.3 will be produce correct results. Newer version will download a different dataset.

# Job

  Your job is to fill the class of APPNP to pass the test. To test locally, simply run:
  
      pytest

  More specifically, change the forward definition of GraphAppnp in the file of "models/appnp.py".

    class GraphAppnp(nn.Module):
        def __init__(self, alpha):
            super().__init__()
            self.alpha = alpha

        def forward(self, x, adj, h):
            '''
            You need to redefine the layer
            '''
            return x

  You may also need to change the hyperparameters at Line53-Line57 in the file of "test_sample.py"

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


## APPNP

   * [APPNP: Predict then Propagate: Graph Neural Networks Meet Personalized Pagerank, ICLR 2019](https://arxiv.org/pdf/1810.05997.pdf)

   * Training

            python3 train.py --model APPNP --hidden 64 --epochs 100 --lr 0.01 --dataset cora
            python3 train.py --model APPNP --hidden 64 --epochs 100 --lr 0.01 --dataset pubmed
            python3 train.py --model APPNP --hidden 64 --epochs 100 --lr 0.01 --dataset citeseer

# Usage

* train.py

      usage: train.py [-h] [--device DEVICE] [--model MODEL] [--data-root DATA_ROOT]
                      [--dataset DATASET] [--seed SEED] [--epochs EPOCHS] [--lr LR]
                      [--weight_decay WEIGHT_DECAY] [--hidden HIDDEN]
                      [--dropout DROPOUT]

      optional arguments:
        -h, --help            show this help message and exit
        --device DEVICE       cpu, cuda:0, cuda:1, etc.
        --model MODEL         GCN, GAT, or APPNP
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
