import pyterrier as pt
import logging
import string
# import re

#FunctionName: 
#   searchTerrierIndex
#Input:
#   query               :   Query of string type 
#   bm25                :   Terrier Retriever object
#   debugMode           :   Boolen variable to enable
#                           debug mode
#                           Default: False
#Output:
#   res                 :   List of relevant docno's
#Description:
#   Searches an already built terrier index 
#Notes:
#   None
def searchTerrierIndex(query, bm25, debugMode=False):
    for p in string.punctuation:
        query = query.replace(p, "")
    # if not re.match("^[a-zA-Z0-9_]*$", query):
    #     if debugMode:
    #         logging.info(f"Query '{query}' contains invalid characters")
    #     return []
    res = bm25.search(query)
    if res.empty:
        if debugMode:
            logging.info(f"Querying '{query}' on Terrier index returned no results")
        return []
    res = list(res.to_dict()["docno"].values())
    return res