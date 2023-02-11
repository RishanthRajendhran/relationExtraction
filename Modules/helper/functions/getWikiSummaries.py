import numpy as np
import logging
import wikipedia

#FunctionName: 
#   getWikiSummaries
#Input:
#   entities    :   List containing MIDs of entities
#   mid2name    :   Dictionary with MIDs as keys and 
#                   list of name mappings as values
#   articles    :   Boolen variable to indicate if articles
#                   need to bee extracted instead of 
#                   summaries
#                   Default: False
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   wikiSummaries    :   Dictionary with titles as keys and article 
#                       summaries as values
#Description:
#   This function is used to extract wikipedia summaries/articles given a set
#   of entities
#Notes:
#   None
def getWikiSummaries(entities, mid2name, articles=False, debugMode=False):
    wikiSummaries = {}
    i = 0
    for entity in entities:
        i += 1
        if entity not in mid2name.keys():
            if debugMode:
                logging.warning(f"Entity with MID: {entity} has no mapping in mid2name!")
            continue 
        if debugMode:
            logging.info(f"Processing entity {i}/{len(entities)}")
        entityNames = mid2name[entity]
        for name in entityNames:
            try: 
                if articles:
                    wikiSumm = wikipedia.page(name, auto_suggest=False).content
                else:
                    wikiSumm = wikipedia.summary(name, auto_suggest=False)
            except:
                continue
            if name in wikiSummaries.keys() and wikiSumm not in wikiSummaries[name]:
                wikiSummaries[name] += " " + wikiSumm
            else:
                wikiSummaries[name] = wikiSumm
    return wikiSummaries