import numpy as np
import logging
import spacy 
import neuralcoref

#FunctionName: 
#   resolveCorefences
#Input:
#   allArticles         :   List of articles
#   debugMode           :   Boolean variable to enable debug mode
#                           Default: False
#Output:
#   allArticlesRes      :   List of coreference resolved articles
#Description:
#   This function is used to resolve coreferences in articles
#Notes:
#   None
def resolveCorefences(allArticles, debugMode=False):
    nlp = spacy.load("en")
    neuralcoref.add_to_pipe(nlp)
    allArticlesRes = []
    i = 0
    for article in allArticles:
        if debugMode:
            logging.info(f"Resolving document {i}/{len(allArticles)}...")
            i += 1
        doc = nlp(article)
        allArticlesRes.append(doc._.coref_resolved)
    return allArticlesRes