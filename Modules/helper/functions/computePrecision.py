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
#   _               :   Average precision across all class labels
#   precPerClass    :   Dictionary of precision for ever class
#Description:
#   This function is used to compute average precision of 
#   output predictions 
#Notes:
#   None
def computePrecision(preds,  targets, debugMode=False):
    vals = np.unique(targets)
    allPrecs =  []
    precPerClass = {}
    for v in vals:
        targetInds = np.where(targets==v)[0]
        predInds = np.where(preds==v)[0]
        corrPreds = len(np.intersect1d(targetInds, predInds))
        if len(predInds):
            curPrec = corrPreds/len(predInds)
        else:
            logging.warning(f"No prediction has the class label {v}")
            continue
        allPrecs.append(curPrec)
        precPerClass[v] = curPrec
        if debugMode:
            logging.info(f"Class {v}: Precision = {round(curPrec*100,2)}%")
    prec = np.mean(allPrecs)
    if debugMode:
        logging.info(f"Average Precision: {prec}")
    return prec, precPerClass