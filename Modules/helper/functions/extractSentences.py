import logging
import nltk

#FunctionName: 
#   extractSentences
#Input:
#   articles     :   List of articles 
#   debugMode    :   Boolean variable to enable debug mode
#                    Default: False
#Output:
#   sentences    :   List of sentences
#Description:
#   This function is used to extract sentences from articles
#Notes:
#   None
def extractSentences(articles, debugMode=False):
    allSents = []
    for art in articles:
        allSents.extend(nltk.tokenize.sent_tokenize(art))
    if debugMode:
        logging.info(f"{len(allSents)} sentences extracted from wiki articles!")
    allSents = list(set(allSents))
    return allSents