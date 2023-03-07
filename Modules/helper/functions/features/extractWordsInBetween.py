import logging
import nltk

#FunctionName: 
#   extractWordsInBetween
#Input:
#   words           :   Sequence of words (List)  
#   word1Pos        :   First word (ending) index
#   word2Pos        :   Second word (starting) index
#   debugMode       :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   wordsInBetween    :   List of words (in-order) in between word1 and word2 in the sequence words 
#Description:
#   This function is used to find words in between two words in a sentence
#Notes:
#   This function filters out stop words
#   Checks only for first occurences of the words
#   The returned list does not include word1 or word2 
def extractWordsInBetween(words, word1Pos, word2Pos, debugMode=False):
    if word1Pos > len(words) or word1Pos < 0:
        logging.error(f"Illegal index for word in words")
        return []
    if word2Pos > len(words) or word2Pos < 0:
        logging.error(f"Illegal index for word in words")
        return []
    stop = set(nltk.corpus.stopwords.words("english"))
    if word1Pos > word2Pos:
        word1Pos, word2Pos = word2Pos, word1Pos
    wordsInBetween = [w for w in range((word1Pos+1), word2Pos) if w not in stop]
    if debugMode:
        logging.info(f"{len(wordsInBetween)} words found in between.")
    return wordsInBetween