import numpy as np
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#FunctionName: 
#   plotConfusion
#Input:
#   preds       :   List of predictions  
#   targets     :   List of target values
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   None      
#Description:
#   This function is used to plot confusion matrix
#Notes:
#   None
def plotConfusion(preds, targets, labels, debugMode=False):
    labels = [l.split("/")[-1] for l in labels]
    confusionMatrix = confusion_matrix(targets, preds)
    if debugMode:
        for i in range(len(confusionMatrix)):
            truePos = confusionMatrix[i][i]
            falsePos = sum(confusionMatrix[i])-truePos
            logging.info(f"Label {labels[i]}:")
            logging.info(f"\tTrue positives: {truePos}")
            logging.info(f"\tFalse positives: {falsePos}")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=labels)
    disp.plot()
    plt.show()
    return