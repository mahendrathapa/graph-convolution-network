import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.init_parameters()

    def init_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support) + self.bias

        return output


class GCN(nn.Module):
    def __init__(self, config):

        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(config.n_features, config.n_hidden_dim)
        self.gc2 = GraphConvolution(config.n_hidden_dim, config.n_class)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
