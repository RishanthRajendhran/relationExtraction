import transformers
import torch
from torch import nn
import numpy as np
import logging

class RelationClassifier(nn.Module):
    def __init__(self, numClasses, pretrainedModel, numAttnHeads, labelEncoderObj=None):
        super().__init__()
        self.numClasses = numClasses
        self.pretrainedModel = pretrainedModel
        self.numAttnHeads = numAttnHeads
        self.bertModel = transformers.BertModel.from_pretrained(self.pretrainedModel)
        self.weightsKey = nn.parameter.Parameter(data=torch.rand(2*self.bertModel.config.hidden_size, 2*self.bertModel.config.hidden_size), requires_grad=True)
        self.weightsQue = nn.parameter.Parameter(data=torch.rand(2*self.bertModel.config.hidden_size, 2*self.bertModel.config.hidden_size), requires_grad=True)
        self.weightsVal = nn.parameter.Parameter(data=torch.rand(2*self.bertModel.config.hidden_size, 2*self.bertModel.config.hidden_size), requires_grad=True)
        self.attn = nn.MultiheadAttention(2*self.bertModel.config.hidden_size, self.numAttnHeads, batch_first=True)
        self.dropOut = nn.Dropout(p=0.3)
        self.outLSTM = nn.LSTM(
            2*self.bertModel.config.hidden_size, 
            1024, 
            bidirectional=True,
            batch_first=True
        )
        # self.outSigmoid = nn.Sigmoid()
        self.out = nn.Linear(2*1024, numClasses)
        self.softMax = nn.Softmax(dim=1)
        self.le = labelEncoderObj
        
    def forward(self, input_ids, attention_mask, entity_pair_inds, return_attn_out=False):
        #Bert does not accept 3D inputs ([batch, sentence, sequence])
        #Convert to 2D inputs ([batch*sentence, sequence])
        oriShape = input_ids.shape
        input_ids = torch.reshape(input_ids, (-1, oriShape[-1]))
        attention_mask = torch.reshape(attention_mask, (-1, oriShape[-1]))
        bertOutput = self.bertModel(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bertOutput["last_hidden_state"] = torch.reshape(bertOutput["last_hidden_state"], oriShape + (self.bertModel.config.hidden_size,))
        bertOutput["pooler_output"] = torch.reshape(bertOutput["pooler_output"], oriShape[:-1] + (self.bertModel.config.hidden_size,))
        subjOut = []
        for i in range(len(bertOutput["last_hidden_state"])):
            curSubjs = []
            for j in range(len(bertOutput["last_hidden_state"][i])):
                curSubjs.append(bertOutput["last_hidden_state"][i][j][entity_pair_inds[i][j][0]].tolist())
            subjOut.append(curSubjs)
        subjOut = torch.tensor(subjOut)
        subjOut = subjOut.to(bertOutput["last_hidden_state"].device)
        objOut = []
        for i in range(len(bertOutput["last_hidden_state"])):
            curObjs = []
            for j in range(len(bertOutput["last_hidden_state"][i])):
                curObjs.append(bertOutput["last_hidden_state"][i][j][entity_pair_inds[i][j][1]].tolist())
            objOut.append(curObjs)
        objOut = torch.tensor(objOut)
        objOut = objOut.to(bertOutput["last_hidden_state"].device)
        catOut = torch.cat((subjOut, objOut), dim=2)
        catOut = catOut.to(bertOutput["last_hidden_state"].device)
        #Add self-attention layer
        keys = catOut @ self.weightsKey
        queries = catOut @ self.weightsQue
        values = catOut @ self.weightsVal 
        attnOut, attnWeights = self.attn(keys, queries, values)
        #[batch, sentence, embedDim]
        #Do a weighted average
        avgOut = torch.sum((attnOut*catOut), dim=1)/oriShape[1]
        #Find representations for <-S> and <-O>
        output = self.dropOut(avgOut)
        output, _ = self.outLSTM(output)
        # output = self.outSigmoid(output)
        output = self.out(output)
        if return_attn_out:
            return self.softMax(output), attnOut
        else:
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

    def resizeTokenEmbeddings(self, numTokens):
        self.bertModel.resize_token_embeddings(numTokens)