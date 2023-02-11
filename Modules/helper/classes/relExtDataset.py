import torch

class RelExtDataset:
    def __init__(self, text, relation, tokenizer, max_len):
        self.text = text 
        self.relation = relation 
        self.tokenizer = tokenizer 
        self.max_len = max_len 

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        example = str(self.text[item])
        encoding = self.tokenizer.encode_plus(
            example,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation = True
        )
        return {
            "text": example,
            "input_ids": encoding["input_ids"].reshape(-1,),
            "attention_mask": encoding["attention_mask"].reshape(-1,),
            "targets": torch.tensor(self.relation[item], dtype=torch.long)
        }