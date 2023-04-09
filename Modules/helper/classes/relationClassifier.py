import torch
import logging
from Modules.helper.classes.treeLSTM import TreeLSTM

class RelationClassifier(torch.nn.Module):
    def __init__(self, numClasses, maxSents, inputSize, hiddenSize, windowSize, dropout, labelEncoderObj=None):
        super().__init__()
        self.numClasses = numClasses
        self.maxSents = maxSents
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.windowSize = windowSize
        self.dropout = dropout
        self.featureSize = self.hiddenSize + (2 + 2 + 2*4*self.windowSize)*self.maxSents
        self.treeLSTM = TreeLSTM(self.inputSize, self.hiddenSize, self.dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.featureSize),
            torch.nn.Linear(self.featureSize, self.numClasses),
            torch.nn.Sigmoid()
        )
        self.le = labelEncoderObj
        self.device = "cpu"
        
    def forward(self, bTrees, rootNodes, featureVectors):
        lstmOut = self.treeLSTM(bTrees)
        lstmOut = list(map(lambda x : x.getHiddenStates(), lstmOut))
        lstmOutTensor = []
        for i in range(len(rootNodes)):
            curOut = []
            for j in range(len(rootNodes[i])):
                curOut.append(lstmOut[i][j][rootNodes[i][j]])
            curOut = torch.stack(curOut)
            curOut = torch.sum(curOut, dim=0)
            lstmOutTensor.append(curOut)
        lstmOut = torch.stack(lstmOutTensor)
        feature = torch.cat((lstmOut, featureVectors),dim=1).to(self.device)
        out = self.classifier(feature)
        return out
        
    def labelToText(self, label):
        if self.le == None:
            logging.error("Label Encoder not initialized!")
            return None
        return self.le.inverse_transform(label)

    def textToLabel(self, text):
        if self.le == None:
            logging.error("Label Encoder not initialized!")
            return None
        return self.le.transform(text)

    def getLabels(self):
        if self.le == None:
            logging.error("Label Encoder not initialized!")
            return None
        return self.le.classes_
    
    def to(self, device):
        self.device = device 
        self = super().to(device)
        self.treeLSTM = self.treeLSTM.to(device)
        return self
    
    def getWindowSize(self):
        return self.windowSize

    def getHiddenSize(self):
        return self.hiddenSize