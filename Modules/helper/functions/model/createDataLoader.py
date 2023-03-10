from torch.utils import data
from helper.classes.relExtDataset import RelExtDataset

#FunctionName: 
#   createDataLoader
#Input:
#   df              :   Pandas Dataframe containing dataset
#   tokenizer       :   Tokenizer object
#   max_len         :   Maximum length of input sequence
#   max_sents       :   Maximum no. of sentences per entity pair
#   batch_size      :   Batch size 
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   _               :   Torch DataLoader
#Description:
#   This function is used to generate a data loader
#Notes:
#   None
def createDataLoader(df, tokenizer, max_len, max_sents, batch_size, debugMode=False):
    ds = RelExtDataset(
        texts = df.texts.to_numpy().reshape(-1,),
        relation = df.relation.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        max_sents=max_sents
    )
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )