import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, skip=True):
        super(GraphConvolution, self).__init__()
        self.skip = skip
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # TODO make fc more efficient via "pack_padded_sequence"
        
        support = torch.bmm(input, self.weight.unsqueeze(
            0).expand(input.shape[0], -1, -1))
        output = torch.bmm(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand(input.shape[0], -1, -1)
        if self.skip:
            output += support

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class Graph(nn.Module):
    
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, dropout):
        super(Graph, self).__init__()
        self.fc_k = nn.Linear(dim_in, dim_hidden)
        self.fc_q = nn.Linear(dim_in, dim_hidden)

        dim_hidden = dim_out if num_layers == 1 else dim_hidden
        self.layers = nn.ModuleList([
            GraphConvolution(dim_in, dim_hidden)
        ])

        for i in range(num_layers - 1):
            dim_tmp = dim_out if i == num_layers-2 else dim_hidden
            self.layers.append(GraphConvolution(dim_hidden, dim_tmp))

        self.dropout = dropout

    
    def build_graph(self, x):
        batch_size, s_len = x.shape[0], x.shape[1]
        emb_k = self.fc_k(x)
        emb_q = self.fc_q(x)
        length = torch.tensor([s_len] * batch_size, dtype=torch.long)

        s = torch.bmm(emb_k, emb_q.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(length, length)):
            s_mask[i][:l_1, :l_2] = 0
        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))

        A = s #F.softmax(s, dim=2)  # [B, t1, t2]
        
        # remove nan from softmax on -inf
        A.data.masked_fill_(A.data != A.data, 0)

        return A
    
    def forward(self, X, A):
        for layer in self.layers:
            X = F.relu(layer(X, A))
            X = F.dropout(X, self.dropout, training=self.training)
        return X
