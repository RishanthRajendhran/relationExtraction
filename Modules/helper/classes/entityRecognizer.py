import torch

class EntityRecognizer(torch.nn.Module):
    pass
    # def __init__(self, numEntTypes, typeToIndex):
    #     super(EntityRecognizer, self).__init__()
    #     self.numEntTypes = numEntTypes
    #     self.typeToIndex = typeToIndex
    #     self.transitions = torch.nn.Parameter(torch.randn(self.numEntTypes, self.numEntTypes))
    #     self.transitions.data[self.typeToIndex["START"], :] = -100000
    #     self.transitions.data[:, self.typeToIndex["END"]] = -100000

    # def forwardAlgorithm(self, features):
    #     alphas = torch.full((1, self.numEntTypes), -1000000)
    #     #In log-space
    #     alphas[0][self.typeToIndex["START"]] = 0

    #     forwardVar = alphas 

    #     for i in range(len(features)):
    #         curAlphas = 
        