import pyterrier as pt
import logging

#FunctionName: 
#   startTerrier
#Input:
#   terrierIndexPath    :   Path where Terrier index is to be stored
#Output:
#   bm25                 :   Terrier Retiever Object
#Description:
#   This function is used to start a Terrier retriever
#Notes:
#   None
def startTerrier(terrierIndexPath, debugMode=False):
    if not pt.started():
        pt.init()
    bm25 = pt.BatchRetrieve(terrierIndexPath, wmodel="BM25")
    if debugMode:
        logging.info("Started Terrier retriever")
    return bm25