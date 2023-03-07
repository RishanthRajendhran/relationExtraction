import torch 
import numpy as np

#FunctionName: 
#   evaluateModel
#Input:
#   model           :   RelClassifier model object
#   dataLoader      :   Torch data loader
#   lossFunction    :   Torch loss function
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
#   This function is used to evaluate a RelClassifier model
#Notes:
#   None
def evaluateModel(model, dataLoader, lossFunction, device, numExamples):
    model = model.eval()

    losses = []
    corrPreds = 0 

    with torch.no_grad():
        for d in dataLoader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            entity_pair_inds = d["entity_pair_inds"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_pair_inds=entity_pair_inds
            )

            _, preds = torch.max(outputs, dim=1)
            loss = lossFunction(outputs, targets)

            corrPreds += torch.sum(preds == targets)
            losses.append(loss.item())
    return corrPreds.double()/numExamples, np.mean(losses)