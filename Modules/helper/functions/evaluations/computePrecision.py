import numpy as np
import logging

#FunctionName: 
#   computePrecision
#Input:
#   preds           :   List of predictions  
#   targets         :   List of target values
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   precs           :   Dictionary containing precisions
#Description:
#   This function is used to compute average precision of 
#   output predictions 
#Notes:
#   None
def computePrecision(preds,  targets, debugMode=False):
    vals = np.unique(targets)
    precPerClass = {}
    macroAvg = 0
    microAvg = 0
    numClasses = 0
    for v in vals:
        targetInds = np.where(targets==v)[0]
        predInds = np.where(preds==v)[0]
        corrPreds = len(np.intersect1d(targetInds, predInds))
        if len(predInds):
            curPrec = corrPreds/len(predInds)
        else:
            logging.warning(f"No prediction has the class label {v}")
            continue
        numClasses += 1
        macroAvg += curPrec
        microAvg += curPrec*len(predInds)
        precPerClass[v] = curPrec
        if debugMode:
            logging.info(f"Class {v}: Precision = {round(curPrec*100,2)}%")
    macroAvg /= numClasses
    microAvg /= len(preds)
    if debugMode:
        logging.info(f"Macro Average Precision: {macroAvg}")
    precs = {
        "macro": macroAvg,
        "micro": microAvg,
        "perClass": precPerClass
    }
    return precs