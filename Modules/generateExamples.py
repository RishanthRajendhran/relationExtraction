from Modules.helper.imports.functionImports import checkFile, extractWords
from Modules.helper.imports.packageImports import argparse, pickle, logging, np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to enable debug mode"
)

parser.add_argument(
    "-log",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-mid2name",
    help="Path to file containing mid2name dictionary",
    default="mid2name.pkl"
)

parser.add_argument(
    "-entities",
    help="Path to file containing entities dictionary",
    default="entities_10.pkl"
)

parser.add_argument(
    "-relations",
    help="Path to file containing relations dictionary",
    default="relations_10.pkl"
)

parser.add_argument(
    "-invertedIndex",
    help="Path to file containing inverted index",
    default="invertedIndex_20000.pkl"
)

parser.add_argument(
    "-negative",
    action="store_true",
    help="Boolean flag to generate negative examples"
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
mid2nameFile = args.mid2name
entitiesFile = args.entities
relationsFile = args.relations
invertedIndexFile = args.invertedIndex
negative = args.negative

checkFile(mid2nameFile, ".pkl")
checkFile(entitiesFile, ".pkl")
checkFile(relationsFile, ".pkl")
checkFile(invertedIndexFile, ".pkl")

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.DEBUG)

with open(mid2nameFile, 'rb') as f:
    mid2name = pickle.load(f)

with open(entitiesFile, 'rb') as f:
    entities = pickle.load(f)

with open(relationsFile, 'rb') as f:
    relations = pickle.load(f)

with open(invertedIndexFile, "rb") as f:
    invertedIndex = pickle.load(f)

def getVals(keyToRetrieve, dictionaryWithVals):
    if keyToRetrieve in dictionaryWithVals.keys():
        return dictionaryWithVals[keyToRetrieve]
    if keyToRetrieve.lower() in dictionaryWithVals.keys():
        return dictionaryWithVals[keyToRetrieve.lower()]
    return None

for reln in relations:
    print(reln)
    for relInst in relations[reln]:
        if relInst[0] not in mid2name.keys():
            if debug:
                logging.warning(f"No mapping found in mid2name for {relInst[0]}")
            continue
        if relInst[1] not in mid2name.keys():
            if debug:
                logging.warning(f"No mapping found in mid2name for {relInst[1]}")
            continue
        entity_1 = mid2name[relInst[0]]
        entity_2 = mid2name[relInst[1]]
        sents_1 = []
        for name in entity_1:
            wordsInName = extractWords(name)
            if len(wordsInName) == 0:
                continue
            reqSents = getVals(wordsInName[0], invertedIndex["index"])
            if reqSents:
                for i in range(1,len(wordsInName)):
                    newSents = getVals(wordsInName[i], invertedIndex["index"])
                    if newSents:
                        reqSents = np.intersect1d(reqSents, newSents)
                    else:
                        reqSents = []
                        break 
                sents_1.extend(reqSents)
        sents_2 = []
        for name in entity_2:
            wordsInName = extractWords(name)
            if len(wordsInName) == 0:
                continue
            reqSents = getVals(wordsInName[0], invertedIndex["index"])
            if reqSents:
                for i in range(1,len(wordsInName)):
                    newSents = getVals(wordsInName[i], invertedIndex["index"])
                    if newSents:
                        reqSents = np.intersect1d(reqSents, newSents)
                    else:
                        reqSents = []
                        break 
                sents_2.extend(reqSents)
        commonSents = np.intersect1d(sents_1, sents_2)
        if len(commonSents) == 0:
            continue
        print(f"{entity_1[0]} x {entity_2[0]}")
        for sent in commonSents:
            print(sent)
    exit(0)


 
        