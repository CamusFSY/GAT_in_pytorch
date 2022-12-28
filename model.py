import torch
import torch.nn as nn
import torch.nn.functional as F
# from layer_old import GraphAttentionLayer
from layer_backup import GraphAttentionLayer
from layers import SpGraphAttentionLayer
from icecream import ic


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions_hid = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_hid):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x: 16*116*128  (batch * node_size * hid_dim)
        # adj: 16*116*116
        x = x.type(torch.float32)
        adj = adj.type(torch.float32)

        x = F.dropout(x, self.dropout, training=self.training)

        # layers
        # input x: 16*116*128  (batch * node_size * hid_dim)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # x:16*116*384 (batch * node_size * [hid_dim*heads])
        # x = F.elu(x)
        x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)
        x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)
        x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)
        x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)
        x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)
        """x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)
        x = torch.cat([att(x, adj) for att in self.attentions_hid], dim=2)  # x:16*116*384
        # x = F.elu(x)"""

        # x = F.dropout(x, self.dropout, training=self.training)
        x, att_out = self.out_att(x, adj)  # x: 16*116*2 (batch * node_size * nclasses)
        x = F.elu(x)

        # graph representation
        x = x.transpose(1, 2)
        node_pooling = torch.nn.AdaptiveMaxPool1d(1)
        x = node_pooling(x)
        x = x.view(x.shape[0], x.shape[1])  # x: 16*2 (batch * nclasses)

        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
