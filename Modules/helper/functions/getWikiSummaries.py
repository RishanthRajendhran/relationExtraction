import numpy as np
import logging
import wikipedia

#FunctionName: 
#   getWikiSummaries
#Input:
#   entities    :   List containing MIDs of entities
#   mid2name    :   Dictionary with MIDs as keys and 
#                   list of name mappings as values
#   debugMode        :   Boolean variable to enable debug mode
#                        Default: False
#Output:
#   wikiSummaries    :   Dictionary with titles as keys and article 
#                       summaries as values
#Description:
#   This function is used to extract wikipedia summaries given a set
#   of entities
#Notes:
#   None
def getWikiSummaries(entities, mid2name, debugMode=False):
    wikiSummaries = {}
    for entity in entities:
        if entity not in mid2name.keys():
            if debugMode:
                logging.warning(f"Entity with MID: {entity} has no mapping in mid2name!")
            continue 
        entityNames = mid2name[entity]
        for name in entityNames:
            titles = wikipedia.search(name, 1)
            if len(titles) == 0:
                continue 
            for title in titles:
                try: 
                    wikiSumm = wikipedia.summary(title, auto_suggest=False)
                except:
                    continue
                if title in wikiSummaries.keys():
                    wikiSummaries[title] += " " + wikiSumm
                else:
                    wikiSummaries[title] = wikiSumm
    return wikiSummaries