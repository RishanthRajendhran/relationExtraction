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
#   rec         :   Average recall across all class labels
#   recPerClass :   Dictionary of recalls for ever class
#Description:
#   This function is used to compute average recall of output 
#   predictions 
#Notes:
#   None
def computeRecall(preds,  targets, debugMode=False):
    vals = np.unique(targets)
    allRecs =  []
    recPerClass = {}
    for v in vals:
        targetInds = np.where(targets==v)[0]
        predInds = np.where(preds==v)[0]
        corrPreds = len(np.intersect1d(targetInds, predInds))
        if len(targetInds):
            curRec = corrPreds/len(targetInds)
        else:
            logging.warning(f"No target label has the class label {v}")
            continue
        allRecs.append(curRec)
        recPerClass[v] = curRec
        if debugMode:
            logging.info(f"Class {v}: Recall = {round(curRec*100,2)}%")
    rec = np.mean(allRecs)
    if debugMode:
        logging.info(f"Average Recall: {rec}")
    return rec, recPerClass