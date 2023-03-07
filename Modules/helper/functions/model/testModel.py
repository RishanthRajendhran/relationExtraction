import torch 
import numpy as np
import logging

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
#   texts           :   List of list of sentences given as examples
#   predictions     :   List of predictions made on the examples
#   predictionProbs :   List of softmax probabilities for predictions
#   trueRelations   :   List of true relations
#   attnOuts        :   List of attnOuts from attention layer
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
    attnOuts = []
    with torch.no_grad():
        for i, d in enumerate(dataLoader):
            logging.debug(f"Processing batch {i}/{len(dataLoader)}")
            curTexts = np.transpose(d["texts"]).tolist()
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            entity_pair_inds = d["entity_pair_inds"].to(device)
            targets = d["targets"].to(device)

            outputs, attnOut = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_pair_inds=entity_pair_inds,
                return_attn_out=True
            )

            _, preds = torch.max(outputs, dim=1)
            texts.extend(curTexts)
            predictions.extend(preds)
            predictionProbs.extend(outputs)
            trueRelations.extend(targets)
            attnOuts.extend(attnOut)

    predictions = torch.stack(predictions).cpu()
    predictionProbs = torch.stack(predictionProbs).cpu()
    trueRelations = torch.stack(trueRelations).cpu()
    attnOuts = torch.stack(attnOuts).cpu()

    return texts, predictions, predictionProbs, trueRelations, attnOuts