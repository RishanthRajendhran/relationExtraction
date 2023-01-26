import logging
import csv


#FunctionName: 
#   extractMappings
#Input:
#   fileName    :   Path to TSV file containing mappings between 
#                   MIDS and wikipedia titles
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   mid2name    :   Dictionary with MIDs as keys and all possible
#                   names as values
#Description:
#   This function is used to extract mappings between MIDS and 
#   wikipedia titles from mappings file
#Notes:
#   fileName is assumed to be a TSV file path
def extractMappings(fileName, debugMode=False):
    with open(fileName, "r") as f:
        TSVreader = csv.reader(f, delimiter="\t")
        data = []
        for line in TSVreader:
            data.append(line)
    mid2name = {}
    for d in data:
        if d[0] not in mid2name.keys():
            mid2name[d[0]] = []
        mid2name[d[0]].append(d[1])

    if debugMode:
        avgAltNames  = 0
        for mid in mid2name.keys():
            avgAltNames += len(mid2name[mid])
        avgAltNames /= len(mid2name)
        logging.info(f"No. of mappings: {len(data)}")
        logging.info(f"No. of MIDs: {len(mid2name)}")
        logging.info(f"Average no. of names per MID: {round(avgAltNames, 2)}")
    
    return mid2name

