import numpy as np
import logging
from datasets import load_dataset

#FunctionName: 
#   getWikiArticles
#Input:
#   numSamples      :   No. of wikipedia articles to extract
#                       Default: All
#Output:
#   wikiArticles    :   Dictionary with titles as keys and article 
#                       content as text
#Description:
#   This function is used to extract wikipedia articles
#Notes:
#   None
def getWikiArticles(numSamples=None):
    wiki = load_dataset("wikipedia", "20220301.en")
    wikiArticles = {}
    l = wiki["train"].__len__()
    if not numSamples:
        numSamples = l
        logging.warning("No numSamples provided, collecting all wiki articles.")
    chosenIndices = np.random.choice(l, size=numSamples, replace=False)
    for article in wiki["train"].select(chosenIndices):
        if article["title"] in wikiArticles:
            title = article["title"]
            print(f"Duplicate {title}")
            break 
        wikiArticles[article["title"]] = article["text"]
    return wikiArticles