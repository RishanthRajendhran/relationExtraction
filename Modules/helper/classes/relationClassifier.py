import transformers
import torch
from torch import nn
import numpy as np

class RelationClassifier(nn.Module):
    def __init__(self, numClasses, pretrainedModel, labelEncoderObj=None):
        super().__init__()
        self.numClasses = numClasses
        self.pretrainedModel = pretrainedModel
        self.bertModel = transformers.BertModel.from_pretrained(self.pretrainedModel)
        self.dropOut = nn.Dropout(p=0.3)
        self.outLSTM = nn.LSTM(
            self.bertModel.config.hidden_size, 
            1024, 
            bidirectional=True,
            batch_first=True
        )
        # self.outSigmoid = nn.Sigmoid()
        self.out = nn.Linear(2*1024, numClasses)
        self.softMax = nn.Softmax(dim=1)
        self.le = labelEncoderObj
        
    def forward(self, input_ids, attention_mask):
        bertOutput = self.bertModel(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.dropOut(bertOutput["pooler_output"])
        output, _ = self.outLSTM(output)
        # output = self.outSigmoid(output)
        output = self.out(output)
        return self.softMax(output)
    
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