from Modules.helper.imports.functionImports import checkFile, extractNERtags, findClosestEntityName, maskWordWithNER
from Modules.helper.imports.packageImports import argparse, pickle, logging, np, difflib

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
    "-examples",
    nargs="+",
    help="List of paths to files containing examples",
    required=True
)

parser.add_argument(
    "-numSamplesPerReln",
    type=int,
    help="No. of samples to pick per relation",
    required=True,
)

parser.add_argument(
    "-out",
    help="Path to sampled examples (extension=.pkl)",
    default="all_examples_2.pkl"
)

parser.add_argument(
    "-pickRelations",
    type=str,
    help="Flag to enable pickRelations mode and specify no. of relations to pick/text file containing relations to pick one per line",
    default=None,
)

parser.add_argument(
    "-append",
    action="store_true",
    help="Boolean flag to aappend to old output"
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
mid2nameFile = args.mid2name
examplesFile = args.examples
outFile = args.out
numSamplesPerReln = args.numSamplesPerReln
pickRelations = args.pickRelations
append = args.append

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else: 
    logging.basicConfig(filemode='w', level=logging.INFO)

for exFile in examplesFile:
    checkFile(exFile, ".pkl")
checkFile(mid2nameFile, ".pkl")

examples = []
for exFile in examplesFile:
    with open(exFile, "rb") as f:
        examples.extend(pickle.load(f))

with open(mid2nameFile, "rb") as f:
    mid2name = pickle.load(f)

relations = []
for ex in examples: 
    if ex["relation"] not in relations:
        relations.append(ex["relation"])
if pickRelations:
    if pickRelations.isdigit():
        numSamples = int(pickRelations)
        if numSamples > len(relations):
            logging.warning(f"Cannot sample more relations ({numSamples}) than available ({len(relations)})!")
            numSamples = len(relations)
        chosenRelns = np.random.choice(len(relations), numSamples, replace=False)
        relations = relations[chosenRelns]
    else:
        checkFile(pickRelations, ".txt")
        with open(pickRelations, 'r') as f:
            relations = list(f.readlines())
            relations = [reln.replace("\n","") for reln in relations]
    

#Group all sentences by entity pairs
entExs = {}
for ex in examples:
    if ex["relation"] not in relations: 
        continue
    if str(ex["entities"]) not in entExs.keys():
        entExs[str(ex["entities"])] = {}
    if ex["relation"] not in entExs[str(ex["entities"])].keys():
        entExs[str(ex["entities"])][ex["relation"]] = []
    entExs[str(ex["entities"])][ex["relation"]].append(ex)  

relExs = {}
for ents in entExs.keys():
    for rel in entExs[ents].keys():
        if rel not in relations: 
            continue
        if rel not in relExs.keys():
            relExs[rel] = {}
        relExs[rel][ents] = entExs[ents][rel]

newExamples = []
outFileName = ".".join(outFile.split(".")[:-1])+".pkl"
if append:
    checkFile(outFileName)
    with open(outFileName,"rb") as f:
        newExamples = pickle.load(f)

for rel in relExs.keys():
    logging.info(f"Processing relation {rel}")
    if len(relExs[rel]) < numSamplesPerReln:
        logging.warning(f"Cannot sample more instances than available.")
    relEntPairs = list(relExs[rel].keys())
    chosenEntPairs = np.random.choice(len(relEntPairs), min(len(relEntPairs), numSamplesPerReln),replace=False)
    entCount = 1
    for cep in chosenEntPairs:
        logging.info(f"\tProcessing entity pair {entCount}/{len(chosenEntPairs)}")
        entCount += 1
        texts = []
        for ex in relExs[rel][relEntPairs[cep]]:
            relation = ex["relation"]
            e1 = ex["entities"][0]
            e2 = ex["entities"][1]
            sentence = ex["doc"]["text"]
            entity1 = None 
            entity2 = None 
            if e1 not in mid2name.keys():
                logging.debug(f"No mapping found for {e1}")
                continue
            if e2 not in mid2name.keys():
                logging.debug(f"No mapping found for {e2}")
                continue
            bestName1, bestMatch1, _, isReliable = findClosestEntityName(sentence, mid2name[e1], debug)
            if not isReliable:
                logging.debug(f"No name found for {e1}")
                continue
            bestName2, bestMatch2, _, isReliable = findClosestEntityName(sentence, mid2name[e2], debug)
            if not isReliable:
                logging.debug(f"No name found for {e2}")
                continue
            if difflib.SequenceMatcher(None, bestMatch1, bestMatch2).ratio() > 0.7:
                logging.debug(f"Best matches found for entity names ({bestMatch1} and {bestMatch2}) are too close!")
                continue
            # #Check if both entities are being recognised by the named-entity recognizer
            NERdefault = "RELATION"
            sentence = maskWordWithNER(sentence, bestMatch1, NERdefault, "S", debug)
            if sentence == None:
                logging.debug("maskWordWithNER returned None on {} and {}. This should not have happened!".format(ex["doc"]["text"], bestMatch1))
                continue
            sentence = maskWordWithNER(sentence, bestMatch2, NERdefault, "O", debug)
            if sentence == None:
                logging.debug("maskWordWithNER returned None on {} and {}. This should not have happened!".format(ex["doc"]["text"], bestMatch2))
                continue
            texts.append(sentence)
        if len(texts) == 0:
            continue
        newEx = {}
        newEx["relation"] = rel
        newEx["entities"] = relEntPairs[cep]
        newEx["entities"] = cep
        newEx["texts"] = texts
        newExamples.append(newEx)
    logging.info(f"Sampled {len(chosenEntPairs)} instances for relation {rel}")

logging.info(f"Sampled and transformed {len(newExamples)} examples")

with open(outFileName, "wb") as f: 
    pickle.dump(newExamples, f)
