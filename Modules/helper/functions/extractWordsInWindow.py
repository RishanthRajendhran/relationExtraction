import logging
import nltk

#FunctionName: 
#   extractWordsInWindow
#Input:
#   words               :   Sequence of words (List)  
#   focusWordStart      :   Starting index of Word under focus
#   focusWordEnd        :   Ending index of Word under focus
#   windowSize          :   Size of the window 
#   debugMode           :   Boolean variable to enable debug mode
#                           Default: False
#Output:
#   wordsInWindowLeft    :   List of words (in-order) within windowSize words to 
#                            the left of focusWord in words
#   wordsInWindowRight   :   List of words (in-order) within windowSize words to 
#                            the right of focusWord in words 
#Description:
#   This function is used to find words in a window around a word in a sentence
#Notes:
#   This function filters out stop words
#   Checks only for first occurences of the word
def extractWordsInWindow(words, focusWordStart, focusWordEnd, windowSize, debugMode=False):
    if focusWordStart < 0 or focusWordStart > len(words):
        logging.error(f"Illegal starting index for word in words")
        return [], []
    if focusWordEnd < 0 or focusWordEnd > len(words):
        logging.error(f"Illegal ending index for word in words")
        return [], []
    if windowSize < 0:
        logging.warning(f"Window size cannot be lower than 1!")
        windowSize = 0
    stop = set(nltk.corpus.stopwords.words("english"))
    wordsInWindowLeft = ["#PAD#"]*windowSize
    winLeft = [w for w in range(max(0,(focusWordStart-windowSize)), focusWordStart) if words[w] not in stop]
    if len(winLeft):
        wordsInWindowLeft[-len(winLeft):] = winLeft
    wordsInWindowRight = ["#PAD#"]*windowSize
    winRight = [w for w in range((focusWordEnd+1), min(len(words),(focusWordEnd+1+windowSize))) if words[w] not in stop]
    if len(winRight):
        wordsInWindowRight[:len(winRight)] = winRight
    if debugMode:
        focusWord = " ".join(words[focusWordStart:(focusWordEnd+1)])
        logging.info(f"Extracted {len(wordsInWindowLeft)} words in left window and {len(wordsInWindowRight)} words in right window of the word {focusWord}!")
    return wordsInWindowLeft, wordsInWindowRight