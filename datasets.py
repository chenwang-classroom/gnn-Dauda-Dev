#!/usr/bin/env python3

import os
import dgl
import torch
from dgl import DGLGraph
from dgl.data import citegrh
from torchvision.datasets import VisionDataset


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
    feat_len, num_class = features.shape[1], data.num_labels
    return adj, features, labels, idx_train, idx_val, idx_test, feat_len, num_class


def graph_collate(batch):
    feature = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    neighbor = [item[2] for item in batch]
    return [feature, labels, neighbor]


class Citation(VisionDataset):
    def __init__(self, root='~/.dgl', name='cora', data_type='train'):
        super(Citation, self).__init__(root)
        self.name = name

        self.download()
        self.features = torch.FloatTensor(self.data.features)
        self.ids = torch.LongTensor(list(range(self.features.size(0))))
        graph = dgl.transform.add_self_loop(DGLGraph(self.data.graph))
        self.src, self.dst = graph.edges()
        self.labels = torch.LongTensor(self.data.labels)
        self.adj = graph.adjacency_matrix()

        if data_type == 'train':
            self.mask = torch.BoolTensor(self.data.train_mask)
        elif data_type == 'val':
            self.mask = torch.BoolTensor(self.data.val_mask)
        elif data_type == 'test':
            self.mask = torch.BoolTensor(self.data.val_mask)
        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        neighbor = self.features[self.dst[self.src==self.ids[self.mask][index]]]
        return self.features[self.mask][index].unsqueeze(-2), self.labels[self.mask][index], neighbor.unsqueeze(-2)
    
    def download(self):
        """Download data if it doesn't exist in processed_folder already."""
        print('Loading {} Dataset...'.format(self.name))
        processed_folder = os.path.join(self.root, self.name)
        os.makedirs(processed_folder, exist_ok=True)
        os.environ["DGL_DOWNLOAD_DIR"] = processed_folder
        data_file = os.path.join(processed_folder, 'data.pt')
        if os.path.exists(data_file):
            self.data = torch.load(data_file)
        else:
            if self.name.lower() == 'cora':
                self.data = citegrh.load_cora()
            elif self.name.lower() == 'citeseer':
                self.data = citegrh.load_citeseer()
            elif self.name.lower() == 'pubmed':
                self.data = citegrh.load_pubmed()
            else:
                raise RuntimeError('Citation dataset name {} wrong'.format(self.name))
            with open(data_file, 'wb') as f:
                torch.save(self.data, data_file)
        self.feat_len, self.num_class = self.data.features.shape[1], self.data.num_labels
