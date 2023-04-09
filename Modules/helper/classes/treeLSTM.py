import torch
import dgl

class TreeLSTM(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, dropout=0.3):
        super(TreeLSTM, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dropout = torch.nn.Dropout(dropout)
        self.Wiuo = torch.nn.Linear(self.inputSize, 3*self.hiddenSize, bias=False)
        self.Uiuo = torch.nn.Linear(self.hiddenSize, 3*self.hiddenSize, bias=False)
        self.biuo = torch.nn.Parameter(torch.zeros(1, 3*self.hiddenSize), requires_grad=True)
        self.Uf = torch.nn.Linear(self.hiddenSize, self.hiddenSize)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tilda = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.Uf(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iuo': nodes.data['iuo'] + self.Uiuo(h_tilda), 'c': c}

    def apply_node_func(self, nodes):
        iuo = nodes.data['iuo'] + self.biuo
        i, u, o = torch.chunk(iuo, 3, 1)
        i, u, o = torch.sigmoid(i), torch.tanh(u), torch.sigmoid(o)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

    def forward(self, batches):
        for b in range(len(batches)):
            curBatch = batches[b]
            curBatch.batchDGLgraph.ndata['iuo'] = self.Wiuo(self.dropout(curBatch.batchDGLgraph.ndata['x']))
            dgl.prop_nodes_topo(curBatch.batchDGLgraph, message_func=self.message_func, reduce_func=self.reduce_func, apply_node_func=self.apply_node_func)
        return batches