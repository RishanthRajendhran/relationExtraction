from torch.utils import data
from helper.classes.relExtDataset import RelExtDataset

#FunctionName: 
#   createDataLoader
#Input:
#   df              :   Pandas Dataframe containing dataset
#   embeddingSize   :   Size of word embeddings used
#   entTypeToInd    :   Dictionary mapping entity types to index
#   posToInd        :   Dictionary mapping part-of-speech tags to index
#   lemmaToInd      :   Dictionary mapping lemma to index 
#   maxSents        :   Maximum no. of sentences per entity pair
#   hiddenSize      :   Size of TreeLSTM hidden representation
#   windowSize      :   Size of context window to consider for building
#                       feature vectors  
#   batchSize       :   Batch Size
#   device          :   Device to load data in (Eg. "cpu", "cuda:0" etc.)
#   debugMode       :   [Deprecated] Boolean flag to enable debug mode
#                       Default: False
#Output:
#   _               :   Torch DataLoader
#Description:
#   This function is used to generate a data loader
#Notes:
#   None
def createDataLoader(df, embeddingSize, entTypeToInd, posToInd, lemmaToInd, maxSents, hiddenSize, windowSize, batchSize, device, debugMode=False):
    ds = RelExtDataset(
        texts = df.cleanedTexts.to_numpy().reshape(-1,),
        relation = df.relation.to_numpy(),
        entities = df.entities.to_numpy(),
        entityPos = df.entityPos.to_list(),
        bottomUpOrders = df.bottomUpOrders.to_list(),
        hierarchies = df.hierarchies.to_list(),
        cleanedTexts = df.cleanedTexts.to_list(),
        allWords = df.allWords.to_list(),
        sentenceVectors = df.sentenceVectors.to_list(),
        allPOS = df.allPOS.to_list(),
        allEntTypes = df.allEntTypes.to_list(),
        allLemmas = df.allLemmas.to_list(),
        allShapes = df.allShapes.to_list(),
        embeddingSize=embeddingSize, 
        entTypeToInd=entTypeToInd, 
        posToInd=posToInd,
        lemmaToInd=lemmaToInd,
        maxSents=maxSents, 
        hiddenSize=hiddenSize, 
        windowSize=windowSize,
        device=device
    )
    return data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        collate_fn=lambda batch: batch,
    )