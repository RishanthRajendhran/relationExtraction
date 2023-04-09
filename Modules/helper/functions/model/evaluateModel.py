import torch 
import numpy as np

#FunctionName: 
#   evaluateModel
#Input:
#   model           :   RelClassifier model object
#   dataLoader      :   Torch data loader
#   lossFunction    :   Torch loss function
#   device          :   Device to train on (Eg. "cpu", "cuda:0")
#   numExamples     :   Total no. of examples used for 
#                       training
#   debugMode       :   [Deprecated] Boolean variable to enable debug mode
#                       Default: False
#Output:
#   _               :   Train accuracy
#   _               :   Train loss
#Description:
#   This function is used to evaluate a RelClassifier model
#Notes:
#   None
def evaluateModel(model, dataLoader, lossFunction, device, numExamples, debugMode=False):
    model = model.eval()

    losses = []
    corrPreds = 0 

    with torch.no_grad():
        for d in dataLoader:
            # texts = []
            bTrees = []
            rootNodes = []
            targets = []
            featureVectors = []
            for i in range(len(d)):
                # texts.append(d[i]["texts"])
                bTrees.append(d[i]["bTree"])
                rootNodes.append(d[i]["rootNodes"])
                targets.append(d[i]["targets"])
                featureVectors.append(d[i]["featureVectors"])
            # texts = torch.tensor(texts).to(device)
            rootNodes = torch.tensor(rootNodes).to(device)
            targets = torch.tensor(targets).to(device)
            featureVectors = torch.tensor(featureVectors).to(device)

            outputs = model(
                bTrees=bTrees,
                rootNodes=rootNodes,
                featureVectors=featureVectors,
            )

            _, preds = torch.max(outputs, dim=1)
            loss = lossFunction(outputs, targets)

            corrPreds += torch.sum(preds == targets)
            losses.append(loss.item())
    return corrPreds.double()/numExamples, np.mean(losses)