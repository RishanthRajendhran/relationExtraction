import logging
import spacy 
import neuralcoref
from Modules.helper.functions.features.findSentContainingToken import findSentContainingToken

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
        if doc._.has_coref:
            sents = [sent for sent in doc.sents]
            sentTexts = [sent.text for sent in doc.sents]
            sentStarts = [sent.start for sent in doc.sents]
            sentEnds = [sent.end for sent in doc.sents]
            sentTokStarts = []
            sentTokEnds = []
            for sent in doc.sents:
                tokStarts = []
                tokEnds = []
                tokOffset = sent[0].idx
                for tok in sent:
                    tokStarts.append(tok.idx-tokOffset)
                    tokEnds.append(tok.idx+len(tok)-tokOffset)
                sentTokStarts.append(tokStarts)
                sentTokEnds.append(tokEnds)
            for cluster in doc._.coref_clusters:
                antecedent = cluster.main
                antSent = findSentContainingToken(sentStarts, sentEnds, antecedent.start, debugMode)
                if antSent == None:
                    if debugMode:
                        logging.warning(f"Could not find sentence containing antecedent!")
                    continue
                #Do not resolve anaphors in the same sentence as the antecedent
                procSents = [antSent]
                for mention in cluster.mentions:
                    if mention.start != antecedent.start and mention.text != antecedent.text:
                        menSent = findSentContainingToken(sentStarts, sentEnds, mention.start, debugMode)
                        if menSent == None:
                            if debugMode:
                                logging.warning(f"Could not find sentence containing mention!")
                            continue
                        #Do not resolve anaphors in an already processed sentence for the same antecedent
                        if menSent not in procSents:
                            procSents.append(menSent)
                            sentStart = sents[menSent].start
                            menStartOffset = mention.start - sentStart
                            menEndOffset = (mention.end-1) - sentStart
                            menStartInd = sentTokStarts[menSent][menStartOffset]
                            menEndInd = sentTokEnds[menSent][menEndOffset]
                            sentTexts[menSent] = sentTexts[menSent][:menStartInd] + antecedent.text + sentTexts[menSent][menEndInd:]
            # allArticlesRes.append(doc._.coref_resolved)
            allArticlesRes.append(" ".join(sentTexts))
        else: 
            allArticlesRes.append(article)
    return allArticlesRes