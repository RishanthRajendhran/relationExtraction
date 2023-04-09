import dgl
import torch

class BatchedTree:
    def __init__(self, trees):
        graphs = []
        for tree in trees:
            graphs.append(tree.dglGraph)
        self.batchDGLgraph = dgl.batch(graphs)

    def getHiddenStates(self):
        graphs = dgl.unbatch(self.batchDGLgraph)
        hiddenStates = []
        maxNumNodes = max([g.num_nodes() for g in graphs])
        for g in graphs:
            hState = g.ndata["h"]
            numNodes, hiddenSize = hState.size()
            if numNodes < maxNumNodes:
                padding = hState.new_zeros(size=(maxNumNodes - numNodes, hiddenSize))
                hState = torch.cat((hState, padding), dim=0)
            hiddenStates.append(hState)
        return torch.stack(hiddenStates)