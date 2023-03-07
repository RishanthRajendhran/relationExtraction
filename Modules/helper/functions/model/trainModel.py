import torch
import numpy as np
import torch.nn as nn

#FunctionName: 
#   trainModel
#Input:
#   model           :   RelClassifier model object
#   dataLoader      :   Torch data loader
#   lossFunction    :   Torch loss function
#   optimizer       :   Torch optimizer
#   device          :   Device to train on 
#                       Choices: ["cpu", "cuda"]
#   scheduler       :   Transformers scheduler
#   numExamples     :   Total no. of examples used for 
#                       training
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   _               :   Train accuracy
#   _               :   Train loss
#Description:
#   This function is used to train a RelClassifier model
#Notes:
#   None
def trainModel(model, dataLoader, lossFunction, optimizer, device, scheduler, numExamples):
    model = model.train()

    losses = []
    corrPreds = 0
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
        #Backwardpropagate the losses
        loss.backward()
        #Avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    return corrPreds.double()/numExamples, np.mean(losses)