import logging

#FunctionName: 
#   findSentContainingToken
#Input:
#   sentStarts          :   List containing index of first token of sentences indexed by sentence number
#   sentEnds            :   List containing index of first token of next sentences indexed by sentence number
#   tokStart            :   Starting index of token to search for
#   debugMode           :   Boolean variable to enable debug mode
#                           Default: False
#Output:
#   _      :   Index of sentence containing the token
#Description:
#   This function is used to find sentence containing spacy type token
#Notes:
#   None
def findSentContainingToken(sentStarts, sentEnds, tokStart, debugMode=False):
    if tokStart < 0:
        logging.error("Token index cannot be negative!")
        return None
    if len(sentStarts) != len(sentEnds):
        logging.error("sentStarts and sentEnds should be of the same length!")
        return None
    if len(sentStarts) == 0:
        logging.error("sentStarts cannot be empty!")
        return None
    return _findSentContainingToken(sentStarts, sentEnds, tokStart, debugMode)

def _findSentContainingToken(sentStarts, sentEnds, tokStart, debugMode=False):
    mid = len(sentStarts)//2
    if len(sentStarts) == 1:
        if sentStarts[mid] <= tokStart and tokStart <= sentEnds[mid]:
            return mid 
        else:
            logging.error("Token index outside document!")
            return None 
    if sentStarts[mid] <= tokStart and tokStart < sentEnds[mid]:
        return mid
    elif sentStarts[mid] > tokStart:
        return _findSentContainingToken(sentStarts[0:mid], sentEnds[0:mid], tokStart)
    else:
        return (mid+1) + _findSentContainingToken(sentStarts[mid+1:], sentEnds[mid+1:], tokStart)