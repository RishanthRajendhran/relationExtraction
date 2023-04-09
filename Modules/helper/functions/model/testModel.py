import torch 
import numpy as np
import logging

#FunctionName: 
#   testModel
#Input:
#   model           :   RelClassifier model object
#   dataLoader      :   Torch data loader
#   device          :   Device to train on (Eg. "cpu", "cuda")
#   debugMode       :   [Deprecated] Boolean variable to enable debug mode
#                       Default: False
#Output:
#   texts           :   List of list of sentences given as examples
#   predictions     :   List of predictions made on the examples
#   predictionProbs :   List of softmax probabilities for predictions
#   trueRelations   :   List of true relations
#Description:
#   This function is used to test a RelClassifier model
#Notes:
#   None
def testModel(model, dataLoader, device, debugMode=False):
    model = model.eval()

    allTexts = []
    predictions = []
    predictionProbs = []
    trueRelations = []
    with torch.no_grad():
        for i, d in enumerate(dataLoader):
            logging.info(f"Processing batch {i}/{len(dataLoader)}")
            texts = []
            bTrees = []
            rootNodes = []
            targets = []
            featureVectors = []
            for i in range(len(d)):
                texts.append(d[i]["texts"])
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
            allTexts.extend(texts)
            predictions.extend(preds)
            predictionProbs.extend(outputs)
            trueRelations.extend(targets)
    return allTexts, predictions, predictionProbs, trueRelations