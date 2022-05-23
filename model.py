import torch as th
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import datetime

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(SAGE, self).__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def load_subtensor(self, nfeat, input_nodes, device):
        """
        Extracts features for a subset of nodes
        """
        if nfeat.is_sparse:
            batch_inputs = [nfeat[node].to_dense() for node in input_nodes]
            batch_inputs = th.stack(batch_inputs)
            batch_inputs = batch_inputs.to(device)
        else:
            batch_inputs = nfeat[input_nodes].to(device)
        return batch_inputs

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)

            # tic = datetime.datetime.now()
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)
            # print("layer %02d | Total sampling time for inference: %s (with %02d num_workers)" % (l, datetime.datetime.now()-tic, num_workers))

            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = self.load_subtensor(x, input_nodes, device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y

class Classifier2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, activation, drop_rate):
        super(Classifier2, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(drop_rate)

        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.out = nn.Linear(hid_dim, out_dim)

    def forward(self, input):
        h1 = self.fc1(input)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)
        out = self.out(h1)
        return out

class GED(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GED, self).__init__()
        self.graphsage = SAGE(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

        self.clf = Classifier2(n_hidden, n_hidden, n_classes, activation, dropout)
        
    def forward(self, blocks, x):
        gnn_emb = self.graphsage(blocks, x)
        out = self.clf(gnn_emb)
        return out

    def inference(self, g, x, device, batch_size, num_workers):
        gnn_emb = self.graphsage.inference(g, x, device, batch_size, num_workers)
        out = self.clf(gnn_emb)
        return out