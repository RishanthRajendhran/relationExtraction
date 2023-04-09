import torch
import regex as re
import logging

class RelExtDataset:
    # def __init__(self, texts, relation, hierarchies, bottomUpOrders, stanzaOuts, tokenizer, max_len, max_sents):
    def __init__(self, texts, relation, tokenizer, max_len, max_sents):
        self.texts = texts 
        self.relation = relation 
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.max_sents = max_sents
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        curSentences = self.texts[item]
        sentInputIDs = []
        sentAttentionMasks = []
        entPairInds = []
        #Pad missing sentences upto self.max_sents
        for i in range(len(curSentences), self.max_sents):
            curSentences.append("")
        #Trim sentences to self.max_sents
        curSentences = curSentences[:self.max_sents]
        #Encode every sentence
        for sentence in curSentences:
            encoding = self.tokenizer.encode_plus(
                sentence,
                max_length=self.max_len,
                add_special_tokens=True,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
                truncation = True
            )
            sentInputIDs.append(encoding["input_ids"].reshape(-1,).tolist())
            sentAttentionMasks.append(encoding["attention_mask"].reshape(-1,).tolist())
            subjInd = 0
            objInd = 0
            if len(sentence):
                subWords = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].reshape(-1,))
                subj = re.compile("<[A-Za-z_]{1,}-S>")
                subjMatches = list(filter(subj.match, subWords))
                if len(subjMatches) == 0:
                    logging.debug(f"Could not find subject entity marker in {subWords}!")
                elif len(subjMatches) > 1:
                    logging.debug(f"More than one subject entity marker found in  {subWords}!")
                else:
                    subjInd =  subWords.index(subjMatches[0])
                obj = re.compile("<[A-Za-z_]{1,}-O>")
                objMatches = list(filter(obj.match, subWords))
                if len(objMatches) == 0:
                    logging.debug(f"Could not find object entity marker in {subWords}!")
                elif len(objMatches) > 1:
                    logging.debug(f"More than one object entity marker found in {subWords}!")
                else:
                    objInd =  subWords.index(objMatches[0])
            entPairInd = (subjInd, objInd)
            entPairInds.append(entPairInd)
        sentInputIDs = torch.tensor(sentInputIDs)
        sentAttentionMasks = torch.tensor(sentAttentionMasks)
        entPairInds = torch.tensor(entPairInds)
        return {
            "texts": curSentences,
            "input_ids": sentInputIDs,
            "attention_mask": sentAttentionMasks,
            "entity_pair_inds": entPairInds,
            "targets": torch.tensor(self.relation[item], dtype=torch.long),
        }