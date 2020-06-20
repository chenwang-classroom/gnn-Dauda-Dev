#!/usr/bin/env python3

# Copyright <2020> <Chen Wang [https://chenwang.site], Carnegie Mellon University>

# Redistribution and use in source and binary forms, with or without modification, are 
# permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of 
# conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list 
# of conditions and the following disclaimer in the documentation and/or other materials 
# provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be 
# used to endorse or promote products derived from this software without specific prior 
# written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.

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
