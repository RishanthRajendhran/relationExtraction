import numpy as np
import logging
from datasets import load_dataset

#FunctionName: 
#   getWikiArticles
#Input:
#   numSamples          :   No. of wikipedia articles to extract
#                           Default: All
#   debugMode           :   Boolean variable to enable debug mode
#                           Default: False
#Output:
#   wikiArticles    :   Dictionary with titles as keys and article 
#                       content as text
#Description:
#   This function is used to extract random wikipedia articles
#Notes:
#   None
def getWikiArticles(numSamples=None, debugMode=False):
    wiki = load_dataset("wikipedia", "20220301.en", cache_dir="")
    wikiArticles = {}
    l = wiki["train"].__len__()
    if not numSamples:
        numSamples = l
        if debugMode:
            logging.warning("No numSamples provided, collecting all wiki articles.")
    chosenIndices = np.random.choice(l, size=numSamples, replace=False)
    for article in wiki["train"].select(chosenIndices):
        if article["title"] in wikiArticles:
            title = article["title"]
            print(f"Duplicate {title}")
            break 
        wikiArticles[article["title"]] = article["text"]
    return wikiArticles