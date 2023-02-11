import pyterrier as pt
import logging

#FunctionName: 
#   buildTerrierIndex
#Input:
#   sentences           :   List of sentences 
#   terrierIndexPath    :   Path where Terrier index is to be stored
#Output:
#   docs                :   List of dictionary containing documents with their docno's
#Description:
#   Builds a terrier index over all sentences 
#Notes:
#   None
def buildTerrierIndex(sentences, terrierIndexPath, debugMode=False):
    docs = []
    for i in range(len(sentences)):
        docs.append({
            "docno": str(i),
            "text": sentences[i]
        })
    #Build Terrier Index  
    if not pt.started():
        pt.init()
    iter_indexer = pt.IterDictIndexer(terrierIndexPath, stemmer="porter", stopwords="terrier")
    _ = iter_indexer.index(docs)
    if debugMode:
        logging.info(f"Successfully built the terrier index at {terrierIndexPath}")
    return docs