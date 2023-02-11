from Modules.helper.imports.functionImports import checkFile, searchTerrierIndex, startTerrier
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
    help="Path to Terrier index folder",
    required=True
)

parser.add_argument(
    "-docs",
    help="Path to documents file (extension='.pkl')",
    required=True
)

parser.add_argument(
    "-negative",
    action="store_true",
    help="Boolean flag to generate negative examples"
)

parser.add_argument(
    "-numExamples",
    type=int,
    help="No. of negative examples to generate",
    default=np.inf
)

parser.add_argument(
    "-out",
    help="Path to store examples (extension=.pkl)",
    default="examples.pkl"
)

parser.add_argument(
    "-allEntities",
    nargs="+",
    help="Negative examples: List of path to all entity files containing entities dictionary",
    default=[]
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
mid2nameFile = args.mid2name
entitiesFile = args.entities
relationsFile = args.relations
terrierIndexPath = args.invertedIndex
docsFile = args.docs
negative = args.negative
outFile = args.out
allEntityFiles = args.allEntities
numExamples = args.numExamples

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if negative and len(allEntityFiles) == 0:
    logging.critical(f"In negative mode, list of paths to all entity files must be provided with -allEntities flag")
    exit(0)

checkFile(mid2nameFile, ".pkl")
checkFile(docsFile, ".pkl")

if negative:
    for entFile in allEntityFiles:
        checkFile(entFile, ".pkl")
else:
    checkFile(entitiesFile, ".pkl")
    checkFile(relationsFile, ".pkl")


if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.DEBUG)

with open(mid2nameFile, 'rb') as f:
    mid2name = pickle.load(f)

if negative:
    entities = {}
    for entFile in allEntityFiles:
        with open(entFile, 'rb') as f:
            curEnts = pickle.load(f)
        for ent in curEnts.keys():
            if ent not in entities.keys():
                entities[ent] = {}
            for rel in curEnts[ent].keys():
                if rel not in entities[ent].keys():
                    entities[ent][rel] = []
                entities[ent][rel].extend(entities[ent][rel])
else:
    with open(entitiesFile, 'rb') as f:
        entities = pickle.load(f)
    with open(relationsFile, 'rb') as f:
        relations = pickle.load(f)

with open(docsFile, "rb") as f:
    docs = pickle.load(f)

bm25 = startTerrier(terrierIndexPath, debug)

if negative:
    allEntities = []
    entRels = {}
    for ent in entities.keys():
        if ent not in allEntities:
            allEntities.append(ent)
        if ent not in entRels.keys():
            entRels[ent] = []
        for rel in entities[ent].keys():
            entRels[ent].extend(entities[ent][rel])
            entRels[ent] = list(set(entRels[ent]))

    #Randomly shuffle list of entities to improve diversity in samples
    allEntities = np.array(allEntities)
    np.random.shuffle(allEntities)
    allEntities = allEntities.tolist()

    examples = []
    for i in range(len(allEntities)):
        for j in range(len(allEntities)):
            if i == j:
                continue
            e1 = allEntities[i]
            e2 = allEntities[j]
            if e1 not in mid2name.keys():
                if debug:
                    logging.info(f"No mapping found for {e1}!")
                continue
            if e2 not in mid2name.keys():
                if debug:
                    logging.info(f"No mapping found for {e2}!")
                continue
            if e2 not in entRels[e1]:
                entity_1 = mid2name[e1]
                entity_2 = mid2name[e2]
                sents_1 = []
                for name in entity_1:
                    reqSents = searchTerrierIndex(name, bm25, debug) 
                    sents_1.extend(reqSents)
                sents_2 = []
                for name in entity_2:
                    reqSents = searchTerrierIndex(name, bm25, debug)  
                    sents_2.extend(reqSents)
                commonSentNos = np.intersect1d(sents_1, sents_2).tolist()
                if len(commonSentNos) == 0:
                    continue
                commonSents = []
                for sNo in commonSentNos:
                    if int(sNo) >= len(docs):
                        logging.critical(f"Terrier returned a document index not found in input docs file!")
                        exit(0)
                    curSent = docs[int(sNo)]
                    curEntPair = (None, None)
                    commonSents.append(curSent)
                    for name in entity_1:
                        if name in curSent:
                            curEntPair[0] = name 
                            break 
                    if curEntPair[0] == None:
                        continue 
                    for name in entity_2:
                        if name in curSent:
                            curEntPair[1] = name 
                            break 
                    if curEntPair[1] == None:
                        continue   
                    examples.append({
                        "relation": "none",
                        "entities": (e1, e2),
                        "entNames": curEntPair,
                        "docno": sNo,
                        "doc": curSent
                    })
                if debug:
                    logging.info(f"\t{entity_1[0]} x {entity_2[0]}")
                    for sent in commonSentNos:
                        logging.info(f"\t\t{sent}")
                if len(examples) > numExamples:
                    break
        if len(examples) > numExamples:
            break 
else:
    examples = []
    for reln in relations:
        if debug:
            logging.info(f"Relation: {reln}")
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
                reqSents = searchTerrierIndex(name, bm25, debug) 
                sents_1.extend(reqSents)
            sents_2 = []
            for name in entity_2:
                reqSents = searchTerrierIndex(name, bm25, debug)  
                sents_2.extend(reqSents)
            commonSentNos = np.intersect1d(sents_1, sents_2).tolist()
            if len(commonSentNos) == 0:
                continue
            commonSents = []
            for sNo in commonSentNos:
                if int(sNo) >= len(docs):
                    logging.critical(f"Terrier returned a document index not found in input docs file!")
                    exit(0)
                curSent = docs[int(sNo)]
                curEntPair = (None, None)
                commonSents.append(curSent)
                for name in entity_1:
                    if name in curSent:
                        curEntPair[0] = name 
                        break 
                if curEntPair[0] == None:
                    continue 
                for name in entity_2:
                    if name in curSent:
                        curEntPair[1] = name 
                        break 
                if curEntPair[1] == None:
                    continue   
                examples.append({
                    "relation": reln,
                    "entities": relInst,
                    "entNames": curEntPair,
                    "docno": sNo,
                    "doc": curSent
                })
            if debug:
                logging.info(f"\t{entity_1[0]} x {entity_2[0]}")
                for sent in commonSentNos:
                    logging.info(f"\t\t{sent}")
    
with open(outFile.split(".")[0]+".pkl","wb") as f:
    pickle.dump(examples, f)  

# with open("relations_test_2.pkl","rb") as f:
#     rel2 = pickle.load(f)
# with open("relations_test_9.pkl","rb") as f:
#     rel9 = pickle.load(f)
# for rel in  rel2.keys():
#     if rel not in rel9.keys():
#         rel9[rel] = []
#     rel9[rel].extend(rel2[rel])
#     rel9[rel] = list(set(rel9[rel]))
# with open("relations_test_11.pkl","wb") as f:
#     pickle.dump(rel9, f)

# with open("entities_test_2.pkl","rb") as f:
#     ent2 = pickle.load(f)
# with open("entities_test_9.pkl","rb") as f:
#     ent9 = pickle.load(f)
# for e in ent2.keys():
#     if e not in ent9.keys():
#         ent9[e] = {}
#     for r in ent2[e].keys():
#         if r not in ent9[e].keys():
#             ent9[e][r] = []
#         ent9[e][r].extend(ent2[e][r])
# with open("entities_test_11.pkl","wb") as f:
#     pickle.dump(ent9, f)