import torch 
import numpy as np

#FunctionName: 
#   testModel
#Input:
#   model           :   RelClassifier model object
#   dataLoader      :   Torch data loader
#   device          :   Device to train on 
#                       Choices: ["cpu", "cuda"]
#   numExamples     :   Total no. of examples used for 
#                       training
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   _               :   Train accuracy
#   _               :   Train loss
#Description:
#   This function is used to test a RelClassifier model
#Notes:
#   None
def testModel(model, dataLoader, device, numExamples, debugMode=False):
    model = model.eval()

    texts = []
    predictions = []
    predictionProbs = []
    trueRelations = []
    with torch.no_grad():
        for d in dataLoader:
            curTexts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            texts.extend(curTexts)
            predictions.extend(preds)
            predictionProbs.extend(outputs)
            trueRelations.extend(targets)

    predictions = torch.stack(predictions).cpu()
    predictionProbs = torch.stack(predictionProbs).cpu()
    trueRelations = torch.stack(trueRelations).cpu()

    return texts, predictions, predictionProbs, trueRelations