from Modules.helper.functions.evaluations.computePrecision import computePrecision
from Modules.helper.functions.evaluations.computeRecall import computeRecall
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
#   scores      :   Dictionary containing f1 scores,
#                   precisions and recall
#Description:
#   This function is used to compute average F1 score of 
#   output predictions 
#Notes:
#   None
def computeF1score(preds, targets, debugMode=False):
    precs = computePrecision(preds, targets, debugMode)
    recs = computeRecall(preds, targets, debugMode)
    if precs["macro"] == 0 and recs["macro"] == 0:
        macroAvg = 0
    else:
        macroAvg = (2*(precs["macro"])*(recs["macro"]))/((precs["macro"])+(recs["macro"]))
    if precs["micro"] == 0 and recs["micro"] == 0:
        microAvg = 0
    else:
        microAvg = (2*(precs["micro"])*(recs["micro"]))/((precs["micro"])+(recs["micro"]))
    f1PerClass = {}
    for clas in precs["perClass"].keys():
        if clas not in recs["perClass"].keys():
            continue
        curPre = precs["perClass"][clas]
        curRec = recs["perClass"][clas]
        if curPre == 0 and curRec == 0:
            f1PerClass[clas] = 0
        else:
            f1PerClass[clas] = (2*curPre*curRec)/(curPre+curRec)
    if debugMode:
        logging.info(f"Macro Average F1 score: {macroAvg}")
    scores = {
        "f1": {
            "macro": macroAvg,
            "micro": microAvg,
            "perClass": f1PerClass 
        },
        "prec": precs,
        "rec": recs
    }
    return scores