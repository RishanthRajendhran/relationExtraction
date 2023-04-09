import dgl 
import torch
import logging

class Tree:
    def __init__(self, hiddenSize, embeddingSize):
        self.dglGraph = dgl.DGLGraph()
        self.hiddenSize = hiddenSize
        self.embeddingSize = embeddingSize
        self.rootNode = None

    def addEdge(self, childID, parentID):
        self.dglGraph.add_edges(childID, parentID)

    def addNode(self, parentID=None, cellInput=None):
        if cellInput is None:
            cellInput = torch.zeros(self.embeddingSize)
        self.dglGraph.add_nodes(1, data={
            "x": cellInput.unsqueeze(0),
            "h": cellInput.new_zeros(size=(1, self.hiddenSize)),
            "c": cellInput.new_zeros(size=(1, self.hiddenSize)),
        })
        newNodeID = self.dglGraph.number_of_nodes()-1
        if parentID is not None:
            self.addEdge(newNodeID, parentID)
        elif self.rootNode == None:
            self.rootNode = newNodeID
        else: 
            logging.error(f"Root node already set! Missing ParentID for new node being added!")
            return None
        return newNodeID
    
    def addNodeBottomUp(self, childrenIDs, cellInput):
        self.dglGraph.add_nodes(1, data={
            "x": cellInput.unsqueeze(0),
            "h": cellInput.new_zeros(size=(1, self.hiddenSize)),
            "c": cellInput.new_zeros(size=(1, self.hiddenSize)),
        })
        newNodeID = self.dglGraph.number_of_nodes()-1
        for childID in childrenIDs:
            self.addEdge(childID, newNodeID)
        return newNodeID
    
    def to(self, device):
        self.dglGraph = self.dglGraph.to(device)
        
    def getRootNode(self):
        return self.rootNode