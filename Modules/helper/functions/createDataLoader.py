from torch.utils import data
from helper.classes.relExtDataset import RelExtDataset

#FunctionName: 
#   createDataLoader
#Input:
#   df              :   Pandas Dataframe containing dataset
#   tokenizer       :   Tokenizer object
#   max_len         :   Maximum length of input sequence
#   batch_size      :   Batch size 
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   _               :   Torch DataLoader
#Description:
#   This function is used to generate a data loader
#Notes:
#   None
def createDataLoader(df, tokenizer, max_len, batch_size, debugMode=False):
    ds = RelExtDataset(
        text = df.text.to_numpy(),
        relation = df.relation.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )