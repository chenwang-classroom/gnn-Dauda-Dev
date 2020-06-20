#!/usr/bin/env python3

import os
import dgl
import torch
from dgl import DGLGraph
from dgl.data import citegrh


def citation(root='~/.dgl', name='cora', device="cpu"):
    name = name.lower()
    print("Loading %s Dataset"%(name))
    processed_folder = os.path.join(root, name.lower())
    os.makedirs(processed_folder, exist_ok=True)
    os.environ["DGL_DOWNLOAD_DIR"] = processed_folder

    if name == 'cora':
        data = citegrh.load_cora()
    elif name == 'citeseer':
        data = citegrh.load_citeseer()
    elif name == 'pubmed':
        data = citegrh.load_pubmed()
    else:
        raise NotImplementedError('Citation %s data name wrong'%(name))

    features = torch.FloatTensor(data.features).to(device)
    graph = dgl.transform.add_self_loop(DGLGraph(data.graph))
    adj = graph.adjacency_matrix().to(device)
    labels = torch.LongTensor(data.labels).to(device)
    idx_train, idx_val, idx_test = torch.BoolTensor((data.train_mask, data.val_mask, data.test_mask)).to(device)
    return adj, features, labels, idx_train, idx_val, idx_test
