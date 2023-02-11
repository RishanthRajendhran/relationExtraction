import numpy as np
import logging

#FunctionName: 
#   computeRecall
#Input:
#   preds       :   List of predictions  
#   targets     :   List of target values
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   recs        :   Dictionary containing recalls
#Description:
#   This function is used to compute average recall of output 
#   predictions 
#Notes:
#   None
def computeRecall(preds,  targets, debugMode=False):
    vals = np.unique(targets)
    recPerClass = {}
    macroAvg = 0
    microAvg = 0
    numClasses = 0
    for v in vals:
        targetInds = np.where(targets==v)[0]
        predInds = np.where(preds==v)[0]
        corrPreds = len(np.intersect1d(targetInds, predInds))
        if len(targetInds):
            curRec = corrPreds/len(targetInds)
        else:
            logging.warning(f"No target label has the class label {v}")
            continue
        numClasses += 1
        macroAvg += curRec
        microAvg += curRec*len(targetInds)
        recPerClass[v] = curRec
        if debugMode:
            logging.info(f"Class {v}: Recall = {round(curRec*100,2)}%")
    macroAvg /= numClasses
    microAvg /= len(targets)
    if debugMode:
        logging.info(f"Macro Average Recall: {macroAvg}")
    recs = {
        "macro": macroAvg,
        "micro": microAvg,
        "perClass": recPerClass
    }
    return recs