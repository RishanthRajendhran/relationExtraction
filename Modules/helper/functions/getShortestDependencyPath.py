import logging
import spacy
import en_core_web_sm
import networkx as nx

#FunctionName: 
#   getShortestDependencyPath
#Input:
#   sentence    :   Sentence (str) 
#   word1       :   First focus word
#   word2       :   Second focus word
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   sdp                 :   Shortest dependency path (as a list of words in the path) 
#   headDepPairs        :   Dictionary of (heads, dependent) pairs
#Description:
#   This function is used to find the shortest dependency path between
#   two words in a sentence
#Notes:
#   None
def getShortestDependencyPath(sentence, word1, word2, debugMode=False):
    sentence = sentence.replace("\n","")
    nlp = en_core_web_sm.load()
    res = nlp(sentence)
    headDepPairs = {}
    for token in res: 
        headDepPairs[(token.head.text, token.text)] = token.dep_
        # if debugMode:
        #     logging.info(f"\t\t{token.head.text}, {token.text}, {token.dep_}")
    edges = []
    for token in res: 
        for child in token.children:
            edges.append(('{}'.format(token.lower_), '{}'.format(child.lower_)))
    graph = nx.Graph(edges)
    sdpl = nx.shortest_path_length(graph, source=word1.lower(),target=word2.lower())
    sdp = nx.shortest_path(graph, source=word1.lower(),target=word2.lower())
    logging.info(sdp)
    return sdp, headDepPairs