import logging
import string
import nltk

#FunctionName: 
#   extractWords
#Input:
#   allSentences     :   List of sentences 
#   debugMode        :   Boolean variable to enable debug mode
#                        Default: False
#Output:
#   words    :   List of words
#Description:
#   This function is used to extract words from sentences
#   excluding stopwords and punctuations
#Notes:
#   None
def extractWords(allSentences, debugMode=False):
    stop = set(nltk.corpus.stopwords.words("english"))
    words = []
    for sent in allSentences:
        curWords = nltk.tokenize.word_tokenize(sent)
        for word in curWords:
            cw = word.replace("\n","")
            if cw not in stop and cw not in string.punctuation:
                words.append(cw)
    words = list(set(words))
    if debugMode:
        logging.info(f"{len(words)} words extracted from sentences!")
    return words