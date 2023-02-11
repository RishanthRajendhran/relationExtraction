from Modules.helper.imports.functionImports import checkFile, extractPOStags, findWordInWords, findClosestEntityName
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


args = parser.parse_args()

debug = args.debug
logFile = args.log
mid2nameFile = args.mid2name
examplesFile = args.examples
outFile = args.out
numSamplesPerReln = args.numSamplesPerReln

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
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

relations = {}
for ex in examples: 
    if ex["relation"] not in relations.keys():
        relations[ex["relation"]] = []
    relations[ex["relation"]].append(ex)

sampledExamples = []
for rel in relations.keys():
    if len(relations[rel]) < numSamplesPerReln:
        logging.warning(f"Cannot sample more instances than available.")
    chosenSamples = np.random.choice(len(relations[rel]), min(len(relations[rel]), numSamplesPerReln),replace=False)
    sampledExamples.extend(np.array(relations[rel])[chosenSamples])
    if debug:
        logging.info(f"Sampled {len(chosenSamples)} instances for relation {rel}")

newExamples = []

for ex in sampledExamples:
    relation = ex["relation"]
    e1 = ex["entities"][0]
    e2 = ex["entities"][1]
    sent = ex["doc"]["text"]
    entity1 = None 
    entity2 = None 
    if e1 not in mid2name.keys():
        if debug:
            logging.info(f"No mapping found for {e1}")
        continue
    if e2 not in mid2name.keys():
        if debug:
            logging.info(f"No mapping found for {e2}")
        continue
    bestName1, bestMatch1, _, isReliable = findClosestEntityName(sent, mid2name[e1], debug)
    if not isReliable:
        if debug:
            logging.info(f"No name found for {e1}")
        continue
    bestName2, bestMatch2, _, isReliable = findClosestEntityName(sent, mid2name[e2], debug)
    if not isReliable:
        if debug:
            logging.info(f"No name found for {e2}")
        continue
    
    newEx = {}
    newEx["relation"] = ex["relation"]
    newEx["entities"] = ex["entities"]
    newEx["entNames"] = (bestName1, bestName2)
    newEx["entPair"] = (bestMatch1, bestMatch2)
    newEx["docno"] = ex["docno"]
    newEx["text"] = ex["doc"]["text"]
    newExamples.append(newEx)

if debug:
    logging.info(f"Sampled and transformed {len(newExamples)} examples")

with open(outFile.split(".")[0]+".pkl", "wb") as f: 
    pickle.dump(newExamples, f)
