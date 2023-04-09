from Modules.helper.imports.functionImports import checkFile
from Modules.helper.functions.features.findWordInWords import findWordInWords
from Modules.helper.imports.packageImports import argparse, pickle, logging, re, spacy
import time

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
    "-examples",
    help="Path to file containing examples",
    required=True
)

parser.add_argument(
    "-out",
    help="Path to preprocessed sampled examples (extension=.pkl)",
    required=True
)

parser.add_argument(
    "-append",
    action="store_true",
    help="Boolean flag to append to old output"
)

parser.add_argument(
    "-start",
    type=int,
    help="Starting index from which examples need to be preprocessed",
    default=-1
)

parser.add_argument(
    "-end",
    type=int,
    help="Ending index until which examples need to be preprocessed",
    default=100000000
)

parser.add_argument(
    "-windowSize",
    type=int,
    help="Size of context window to consider",
    default=3
)

args = parser.parse_args()

debug = args.debug
logFile = args.log
examplesFile = args.examples
outFile = args.out
append = args.append
startInd = args.start 
endInd = args.end
windowSize = args.windowSize

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else: 
    logging.basicConfig(filemode='w', level=logging.INFO)

checkFile(examplesFile, ".pkl")
with open(examplesFile, "rb") as f:
    examples = pickle.load(f)

if startInd < 0:
    startInd = 0
elif startInd > len(examples):
    logging.error("Starting index ({}) cannot be greater than number of examples ({})!".format(startInd, len(examples)))
    exit(0)
if endInd > len(examples):
    endInd = len(examples)
elif endInd < 0:
    logging.error("Ending index ({}) cannot be smaller than zero!".format(startInd))
    exit(0)

logging.info("Examples file: {}".format(examplesFile))
logging.info("Total no. of examples: {}".format(len(examples)))
logging.info("Starting Index: {}".format(startInd))
logging.info("Ending Index: {}".format(endInd))
logging.info("No. of examples to be preprocessed: {}".format(endInd-startInd))

examples = examples[startInd:endInd]

outFileName = ".".join(outFile.split(".")[:-1])+".pkl"
if append:
    checkFile(outFileName)
    with open(outFileName,"rb") as f:
        examplesToAppend = pickle.load(f)
    logging.info("Found {} already preprocessed examples.".format(len(examplesToAppend)))
nlp = spacy.load("en_core_web_lg")
exNum = -1
startExecutionTime = time.time()
for ex in examples:
    exNum += 1
    logging.debug("Preprocessing example {}/{}".format(exNum, len(examples)))
    sentences = ex["texts"]
    entities = []
    entityPos = []
    bottomUpOrders = []
    hierarchies = []
    cleanedTexts = []
    allWords = []
    allEntTypes = []
    allPOS = []
    allLemmas = []
    allShapes = []
    sentenceVectors = []
    for s in sentences:
        if "<RELATION-S>" not in s or "</RELATION-S>" not in s or "<RELATION-O>" not in s or "</RELATION-O>" not in s:
            continue
        entity1 = s[s.find("<RELATION-S>")+len("<RELATION-S>"):s.find("</RELATION-S>")]
        entity2 = s[s.find("<RELATION-O>")+len("<RELATION-O>"):s.find("</RELATION-O>")]

        sentence = re.sub("</?RELATION-[S|O]>","",s)
        doc = nlp(sentence)
        words = []
        posTags = []
        entTypes = {}
        lemmas = []
        shapes = []
        sentenceVector = []
        hierarchy = {
            -1: {
                "parent": None,
                "dep": None,
                "children": set()
            }
        }
        rootNode = -1
        for token in doc: 
            words.append(token.text)
            sentenceVector.append(token.vector)
            posTags.append(token.pos_)
            lemmas.append(token.lemma_)
            shapes.append(token.shape_)
            nodeID = token.i
            parentID = token.head.i
            dep = token.dep_
            if token.head.dep_ == "ROOT":
                parentID = -1
            if nodeID not in hierarchy.keys():
                hierarchy[nodeID] = {
                    "parent": None,
                    "dep": None,
                    "children": set()
                }
            hierarchy[nodeID]["parent"] = parentID
            hierarchy[nodeID]["dep"] = dep
            if parentID not in hierarchy.keys():
                hierarchy[parentID] = {
                    "parent": None,
                    "dep": None,
                    "children": set()
                }
            hierarchy[parentID]["children"].add(nodeID)
        bottomUpOrder = []
        nodesUnderConsideration = {rootNode}
        while(len(nodesUnderConsideration)): 
            bottomUpOrder.append(list(nodesUnderConsideration))
            nextLevelNodes = set()
            for node in nodesUnderConsideration:
                nextLevelNodes.update(hierarchy[node]["children"])
            nodesUnderConsideration = nextLevelNodes
        bottomUpOrder = bottomUpOrder[::-1]

        for span in doc.ents:
            startInd, endInd = None, None
            for token in span:
                if token.text not in ['\"',"\'","\\","\n","\r","\t","\b","\f"]:
                    endInd = token.i
                    if startInd == None:
                        startInd = token.i 
            if startInd and endInd:
                entTypes[startInd] = {
                    "start": startInd, 
                    "end": endInd,
                    "text": span.text,
                    "type": span.label_
                }

        ent1StartInd, ent1EndInd = findWordInWords(words, entity1)
        if ent1StartInd == -1 or ent1EndInd == -1:
            continue
        ent2StartInd, ent2EndInd = findWordInWords(words, entity2)
        if ent2StartInd == -1 or ent2EndInd == -1:
            continue

        entities.append((entity1, entity2))
        entityPos.append(((ent1StartInd, ent1EndInd), (ent2StartInd, ent2EndInd)))
        bottomUpOrders.append(bottomUpOrder)
        hierarchies.append(hierarchy)
        cleanedTexts.append(sentence)
        allWords.append(words)
        sentenceVectors.append(sentenceVector)
        allPOS.append(posTags)
        allEntTypes.append(entTypes)
        allLemmas.append(lemmas)
        allShapes.append(shapes)
    ex.update({
        "entities": entities,
        "entityPos": entityPos,
        "bottomUpOrders": bottomUpOrders,
        "hierarchies": hierarchies,
        "cleanedTexts": cleanedTexts,
        "allWords": allWords,
        "sentenceVectors": sentenceVectors,
        "allPOS": allPOS,
        "allEntTypes": allEntTypes,
        "allLemmas": allLemmas,
        "allShapes": allShapes,
    })
endExecutionTime = time.time()
logging.info("Time taken: {}".format(endExecutionTime - startExecutionTime))

logging.info(f"Preprocessed {len(examples)} sampled examples")

if append:
    examples.extend(examplesToAppend)

with open(outFileName, "wb") as f: 
    pickle.dump(examples, f)
