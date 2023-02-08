from Modules.helper.functions.computePrecision import computePrecision
from Modules.helper.functions.computeRecall import computeRecall
import numpy as np
import logging

#FunctionName: 
#   computeF1score
#Input:
#   preds       :   List of predictions  
#   targets     :   List of target values
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   f1          :   Average f1 score across all class 
#                   labels
#Description:
#   This function is used to compute average F1 score of 
#   output predictions 
#Notes:
#   None
def computeF1score(preds, targets, debugMode=False):
    prec, precPerClass = computePrecision(preds, targets, debugMode)
    rec, recPerClass = computeRecall(preds, targets, debugMode)
    f1 = (2*prec*rec)/(prec+rec)
    f1PerClass = {}
    for clas in precPerClass.keys():
        if clas not in recPerClass.keys():
            continue
        p = precPerClass[clas]
        r = recPerClass[clas]
        if p == 0 and r == 0:
            f1PerClass[clas] = 0
        else:
            f1PerClass[clas] = (2*p*r)/(p+r)
        if debugMode:
            logging.info("Class {}: F1 score = {}".format(clas, f1PerClass[clas]))
    if debugMode:
        logging.info(f"Average F1 score: {f1}")
    return f1, f1PerClass