import logging
import nltk

#FunctionName: 
#   extractPOStags
#Input:
#   sentence    :   Sentence (str) 
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   POStags    :   List of (words, POS tags) tuples 
#   words      :   List of words in sentence 
#Description:
#   This function is used to find POS tags for words in a sentence
#Notes:
#   None
def extractPOStags(sentence, debugMode=False):
    sentence = sentence.replace("\n","")
    curWords = nltk.tokenize.word_tokenize(sentence)
    POStags = nltk.pos_tag(curWords)
    if debugMode:
        logging.info(f"{len(POStags)} words and their POS tags extracted from sentences!")
    return (POStags, curWords)