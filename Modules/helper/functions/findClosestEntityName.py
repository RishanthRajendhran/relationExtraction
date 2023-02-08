from Modules.helper.functions.matchWordInSentence import matchWordInSentence
import nltk

#FunctionName: 
#   findClosestEntityName
#Input:
#   sentences       :   A sentence (str)
#   names           :   List of names for entity to search for
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   bestName        :   Name corresponding to bestMatch
#   bestMatch       :   Closest match to word in sentence
#   bestRecall      :   Recall of the closest match wrt  
#                       bestName
#   isReliable      :   Boolean value indicating if the match
#                       is reliable (and not spurious)
#Description:
#   This function is used to find the name mention of an entity
#   in a sentence based on recall
#Notes:
#   None
def findClosestEntityName(sentence, names, debugMode=False):
    bestRecall, bestName, bestMatch = 0, "", ""
    for name in names:
        matchedPhrase, recall = matchWordInSentence(sentence, name, debugMode)
        if recall > bestRecall:
            bestRecall = recall 
            bestName = name
            bestMatch = matchedPhrase
    if len(bestName):
        wordsInBestName = nltk.tokenize.word_tokenize(bestName)
        n = len(wordsInBestName)
        #If not even two words (in two words or longer names) are extracted, probably not a reliable sample
        isReliable = bestRecall >= (min(2, n)/n)
    else:
        isReliable = False
    return bestName, bestMatch, bestRecall, isReliable