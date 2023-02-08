import logging
import nltk

#FunctionName: 
#   findWordInWords
#Input:
#   words               :   Sequence of words (List)  
#   focusWord           :   Word under focus
#   debugMode           :   Boolean variable to enable debug mode
#                           Default: False
#Output:
#   wordStart           :   Starting index of word in words 
#   wordEnd             :   Ending index of word in words 
#Description:
#   This function is used to locate a word/phrase in a list of words
#Notes:
#   None
def findWordInWords(words, focusWord, debugMode=False):
    if focusWord not in " ".join(words):
        logging.error(f"{focusWord} not present in the list of words!")
        return -1, -1
    wordsInWord = nltk.tokenize.word_tokenize(focusWord)
    sInd = None
    indices = [i for i, x in enumerate(words) if x == wordsInWord[0]]
    for i in indices:
        sInd = i
        corInd = True 
        for w in range(1, len(wordsInWord)):
            if words[sInd+w] != wordsInWord[w]:
                corInd = False 
                break
        if corInd:
            break 
        if not corInd:
            sInd = None
    if sInd == None:
        logging.error(f"Could not find {focusWord} in the list of words!")
        return -1, -1 
    return sInd, sInd+len(wordsInWord)-1 
 