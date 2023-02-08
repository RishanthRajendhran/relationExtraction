import transformers
import torch
from torch import nn
import numpy as np

class RelationClassifier(nn.Module):
    def __init__(self, numClasses, pretrainedModel):
        super().__init__()
        self.numClasses = numClasses
        self.pretrainedModel = pretrainedModel
        self.bertModel = transformers.BertModel.from_pretrained(self.pretrainedModel)
        self.dropOut = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bertModel.config.hidden_size, numClasses)
        self.softMax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        bertOutput = self.bertModel(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.dropOut(bertOutput["pooler_output"])
        output = self.out(output)
        return self.softMax(output)

    def train(self, dataLoader, lossFunction, optimizer, device, scheduler, numExamples):
        self.model = self.model.train()

        losses = []
        corrPreds = 0
        for d in dataLoader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = lossFunction(outputs, targets)

            corrPreds += torch.sum(preds == targets)
            losses.append(loss.item())
            #Backwardpropagate the losses
            loss.backward()
            #Avoid exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return corrPreds.double()/numExamples, np.mean(losses)

    def evaluate(self, dataLoader, lossFunction, device, numExamples):
        self.model = self.model.eval()

        losses = []
        corrPreds = 0 

        with torch.no_grad():
            for d in dataLoader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs, dim=1)
                loss = lossFunction(outputs, targets)

                corrPreds += torch.sum(preds == targets)
                losses.append(loss.item())
        return corrPreds.double()/numExamples, np.mean(losses)