from Modules.helper.imports.functionImports import checkFile, createDataLoader, testModel, computeF1score, plotConfusion, extractNERtags, findWordInWords
from Modules.helper.imports.packageImports import argparse, pickle, logging, np, plt, pd, torch, itertools, spacy, nltk
from Modules.helper.classes.relationClassifier import RelationClassifier
from Modules.helper.classes.tree import Tree 
from Modules.helper.classes.batchedTree import BatchedTree
from Modules.helper.classes.treeLSTM import TreeLSTM
from Modules.helper.classes.colours import Colours

import warnings
warnings.filterwarnings("ignore")

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
    "-model",
    help="Path to file containing RelClassifier model (extension= .pt)",
    default="model.pt"
)

parser.add_argument(
    "-test",
    help="Path to file containing test examples (extension=.pkl)",
    default="all_examples_test_2.pkl"
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size for dataloader",
    default=8
)

parser.add_argument(
    "-live",
    action="store_true",
    help="Boolean flag to enable live demo mode"
)

parser.add_argument(
    "-examples",
    type=str,
    nargs="+",
    help="Example sentences/paragraph (Sentences will be extracted from the paragraph) to test the model on in live demo mode",
    default=None
)

parser.add_argument(
    "-confusion",
    action="store_true",
    help="Boolean flag to plot confusion matrix after evaluation"
)

parser.add_argument(
    "-maxSents",
    type=int,
    help="Maximum no. of sentences per examples",
    default=5
)

parser.add_argument(
    "-histogram",
    action="store_true",
    help="Boolean flag to show histogram of examples"
)


parser.add_argument(
    "-entities",
    type=str,
    nargs="+",
    help="List of entities to consider in examples passed in live demo mode with entityPairs flag enabled",
    default=None
)

parser.add_argument(
    "-savePredictions",
    action="store_true",
    help="Boolean flag to save predictions"
)

parser.add_argument(
    "-printCorPredictions",
    action="store_true",
    help="Boolean flag to print correct predictions"
)

parser.add_argument(
    "-printMisPredictions",
    action="store_true",
    help="Boolean flag to print wrong predictions"
)

parser.add_argument(
    "-baseline",
    choices=["majority", "random"],
    help="Test Baseline models",
    default=None
)

parser.add_argument(
    "-embeddingSize",
    type=int,
    help="Size of embeddings in sentence vectors (from Spacy)",
    default=300
)

parser.add_argument(
    "-entTypeToInd",
    help="Path to .pkl file containg mapping between NER types and integers",
    default="entTypeToInd.pkl"
)

parser.add_argument(
    "-posToInd",
    help="Path to .pkl file containg mapping between POS tags and integers",
    default="posToInd.pkl"
)

parser.add_argument(
    "-lemmaToInd",
    help="Path to .pkl file containg mapping between lemma and integers",
    default="lemmaToInd.pkl"
)

parser.add_argument(
    "-load",
    help="Path to file containing model to load",
    default="fullModel.pt"
)

# parser.add_argument(
#     "-windowSize",
#     type=int,
#     help="Size of context window to consider",
#     default=3
# )

# parser.add_argument(
#     "-hiddenSize",
#     type=int,
#     help="Size of hidden representation in TreeLSTM",
#     default=768
# )

args = parser.parse_args()

debug = args.debug
logFile = args.log
modelFile = args.model
testFile = args.test
batchSize = args.batchSize
live = args.live
examples = args.examples
confusion = args.confusion
maxSents = args.maxSents
histogram = args.histogram
entities = args.entities
savePredictions = args.savePredictions
printCorPredictions = args.printCorPredictions
printMisPredictions = args.printMisPredictions
baseline = args.baseline 
embeddingSize = args.embeddingSize
entTypeToIndFile = args.entTypeToInd
posToIndFile = args.posToInd
lemmaToIndFile = args.lemmaToInd
loadModel = args.load
# windowSize = args.windowSize
# hiddenSize = args.hiddenSize

if logFile:
    checkFile(logFile, ".txt")
if not live:
    checkFile(testFile, ".pkl")
checkFile(loadModel, ".pt")
checkFile(entTypeToIndFile, ".pkl")
checkFile(posToIndFile, ".pkl")
checkFile(lemmaToIndFile, ".pkl")

with open(entTypeToIndFile, "rb") as f:
    entTypeToInd = pickle.load(f)

with open(posToIndFile, "rb") as f:
    posToInd = pickle.load(f)

with open(lemmaToIndFile, "rb") as f:
    lemmaToInd = pickle.load(f)

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if live: 
    logging.basicConfig(level=logging.ERROR)
else:
    with open(testFile, "rb") as f:
        test = pickle.load(f)
    df = pd.DataFrame(test)
    if histogram:
        vals, counts = np.unique(df["relation"].to_numpy(),return_counts=True)  
        vals = [v.split("/")[-1] for v in vals]
        plt.bar(vals, counts)
        plt.xlabel("Relation")
        plt.ylabel("No. of instances")
        plt.title("No. of instances in per relation in train set")
        # plt.show()
        plt.savefig("{}_relExs.png".format(testFile.split(".")[0]))
        plt.clf()
        relNames = df["relation"].unique().tolist()
        relSentCounts = {}
        for rel in relNames:
            relSentCounts[rel.split("/")[-1]] = 0
            dfRel = df[df["relation"] == rel]
            for _, row in dfRel.iterrows():
                relSentCounts[rel.split("/")[-1]] += len(row["texts"])
            relSentCounts[rel.split("/")[-1]] /= len(dfRel)
            relSentCounts[rel.split("/")[-1]] = round(relSentCounts[rel.split("/")[-1]], 2)
        plt.bar(relSentCounts.keys(), relSentCounts.values())
        plt.xlabel("Relation")
        plt.ylabel("Avg. sentences per example instance")
        plt.title("Avg. sentences per example instance per relation in train set")
        # plt.show()
        plt.savefig("{}_relSents.png".format(testFile.split(".")[0]))
        plt.clf()

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

#To tackle error while trying to load a model trained on GPU in a CPU
if torch.cuda.is_available():
    model = torch.load(modelFile)
else:
    model = torch.load(modelFile, map_location={'cuda:0': 'cpu'})
model = model.to(device)

hiddenSize = model.getHiddenSize()
windowSize = model.getWindowSize()

if not live: 
    df['relation'] = model.textToLabel(df['relation'])

if not live: 
    testDataLoader = createDataLoader(df, embeddingSize, entTypeToInd, posToInd, lemmaToInd, maxSents, hiddenSize, windowSize, batchSize, device, debug)
if not live:    
    logging.info(f"File: {testFile}")
if baseline:
    if live:
        logging.error("Cannot test baseline models in live demo mode!")
        exit(0)
    if baseline == "majority":
        labels, counts = np.unique(df.relation.to_numpy(),return_counts=True)
        logging.info("Baseline: Majority")
        majorityInd = np.argmax(counts)
        logging.info("\tMajority prediction: {}".format(model.labelToText([labels[majorityInd]])[0]))
        logging.info("\tAccuracy: {}%".format(round(counts[majorityInd]/sum(counts)*100,2)))
        logging.info("\tPrecision (for majority class): {}".format(round(counts[majorityInd]/sum(counts),2)))
        logging.info("\tRecall (for majority class): 1")
        logging.info("\tF1 Score (for majority class): {}".format(round(counts[majorityInd]/sum(counts),2)))
    elif baseline == "random":
        labels = np.unique(df.relation.to_numpy())
        preds = np.random.choice(labels, df.shape[0], replace=True)
        scores = computeF1score(preds, df.relation.to_numpy(), debug)
        logging.info("Baseline: Uniformly at random")
        logging.info("\tAccuracy: {}%".format(round((np.sum(np.array(preds) == df.relation.to_numpy())/len(preds))*100,2)))
        logging.info("\tPrecision:")
        logging.info("\t\tMacro average: {}".format(round(scores["prec"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["prec"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["prec"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(model.labelToText([clas])[0], round(scores["prec"]["perClass"][clas],2)))
        logging.info("\tRecall:")
        logging.info("\t\tMacro average: {}".format(round(scores["rec"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["rec"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["rec"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(model.labelToText([clas])[0], round(scores["rec"]["perClass"][clas],2)))
        logging.info("\tF1 Score:")
        logging.info("\t\tMacro average: {}".format(round(scores["f1"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["f1"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["f1"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(model.labelToText([clas])[0], round(scores["f1"]["perClass"][clas],2)))
else:
    if live: 
        if examples == None:
            logging.error("Need to provide an example sentences using the -examples flag when in live mode!")
            exit(0)
        newExamples = []
        for ex in examples:
            newExamples.extend(list(nltk.tokenize.sent_tokenize(ex)))
        examples = newExamples
        entitiesUnderConsideration = []
        if entities:
            for ent in entities:
                found = False
                for ex in examples:
                    if ent in ex:
                        entitiesUnderConsideration.append(ent)
                        found = True
                        break
                if not found:
                    logging.warning(f"{ent} not present in any example!")
            entitiesUnderConsideration = list(set(entitiesUnderConsideration))
            if len(entitiesUnderConsideration) < 2:
                logging.error("Not enough distinct named entities in the given examples!")
                exit(0)
        else:
            for ex in examples:
                ner = extractNERtags(ex, debug)
                for neText, _ in ner:
                    entitiesUnderConsideration.append(neText)
            entitiesUnderConsideration = list(set(entitiesUnderConsideration))
            if len(entitiesUnderConsideration) < 2:
                logging.error("Not enough distinct named entities recognized by Spacy NER in the given examples!")
                exit(0)
        logging.info("Inputs:")
        for ex in examples:
            entsInEx = []
            entsPosInEx = []
            for ent in entitiesUnderConsideration:
                if ent in ex: 
                    entsInEx.append(ent)
                    entsPosInEx.append(ex.index(ent))
            if len(entsInEx):
                entsPos = list(zip(entsPosInEx, np.arange(len(entsPosInEx))))
                entsPos.sort()
                exHighlighted = ex[:entsPos[0][0]]
                for eInd in range(len(entsPos)):
                    exHighlighted +=  Colours.BOLD + entsInEx[entsPos[eInd][1]] + Colours.ENDC
                    if eInd == (len(entsPos)-1):
                        exHighlighted += ex[entsPos[eInd][0]+len(entsInEx[entsPos[eInd][1]]):]
                    else:
                        exHighlighted += ex[entsPos[eInd][0]+len(entsInEx[entsPos[eInd][1]]):entsPos[eInd+1][0]]
            else: 
                exHighlighted = ex
            logging.info("\t{}".format(exHighlighted))
        # spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
        for pair in itertools.permutations(entitiesUnderConsideration,2):
            #Determine which sentences contain the entity pair of interest
            curSentences = []
            for ex in examples:
                if pair[0] in ex and pair[1] in ex:
                        curSentences.append(ex)
            #If no sentence contains both the entities, ignore the pair
            if len(curSentences) == 0:
                continue
            entityPos = []
            bottomUpOrders = []
            hierarchies = []
            cleanedTexts = []
            allWords = []
            sentenceVectors = []
            allPOS = []
            allEntTypes = []
            allLemmas = []
            entities = []
            #From preprocessExamples
            sentsToRem = []
            for sentence in curSentences:
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

                ent1StartInd, ent1EndInd = findWordInWords(words, pair[0])
                if ent1StartInd == -1 or ent1EndInd == -1:
                    sentsToRem.append(sentence)
                    continue
                ent2StartInd, ent2EndInd = findWordInWords(words, pair[1])
                if ent2StartInd == -1 or ent2EndInd == -1:
                    sentsToRem.append(sentence)
                    continue

                entities.append(pair)
                entityPos.append(((ent1StartInd, ent1EndInd), (ent2StartInd, ent2EndInd)))
                bottomUpOrders.append(bottomUpOrder)
                hierarchies.append(hierarchy)
                cleanedTexts.append(sentence)
                allWords.append(words)
                sentenceVectors.append(sentenceVector)
                allPOS.append(posTags)
                allEntTypes.append(entTypes)
                allLemmas.append(lemmas)
            
            for sent in sentsToRem:
                curSentences.remove(sent)

            #From RelExtDataset
            curEntities = entities
            curEntTypes = allEntTypes
            curPOS = allPOS
            curLemmas = allLemmas
            curEntityPos = entityPos
            rootNodes = []
            trees = []
            featureVectors = []

            if len(curEntityPos) == 0:
                continue
        
            #Pad missing sentences upto maxSents
            diffToPad = maxSents - len(curSentences)
            curSentences.extend([""]*diffToPad)
            curEntities.extend([("","")]*diffToPad)
            curEntTypes.extend([{}]*diffToPad)
            curPOS.extend([[]]*diffToPad)
            curLemmas.extend([[]]*diffToPad)
            curEntityPos.extend([((-1,-1), (-1,-1))]*diffToPad)

            #Encode every sentence
            for i in range(maxSents):
                #Build Tree
                newTree = Tree(hiddenSize, embeddingSize)
                if len(curSentences[i])==0:
                    newTree.addNode()
                else:
                    bottomUpOrder = bottomUpOrders[i]
                    hierarchy = hierarchies[i]
                    spacyPosToNodeID = {}
                    topDownOrder = bottomUpOrder[::-1]
                    for b in range(len(topDownOrder)):
                        for c in range(len(topDownOrder[b])):
                            parentPos = hierarchy[topDownOrder[b][c]]["parent"]
                            if parentPos:
                                if parentPos not in spacyPosToNodeID.keys():
                                    logging.error("Could not find {} in spacyPosToNodeID! This should not have happened!".format(parentPos))
                                    print(topDownOrder)
                                    exit(0)
                                parentID = spacyPosToNodeID[parentPos]
                            else: 
                                parentID = parentPos
                            curNodePos = topDownOrder[b][c]
                            if parentID == None:
                                curNodeID = newTree.addNode()
                            else:
                                curNodeID = newTree.addNode(parentID, torch.tensor(sentenceVectors[i][curNodePos], dtype=torch.float32))
                            if curNodePos in spacyPosToNodeID.keys():
                                logging.error("Word with pos: {} already seen in topDownOrder! This should not have happened!".format(curNodePos))
                                exit(0)
                            spacyPosToNodeID[curNodePos] = curNodeID
                #Build feature vector 
                featureVector = []
                #Entity Types
                ent1Pos = curEntityPos[i][0]
                ent2Pos = curEntityPos[i][1]
                #RELATION-S
                if ent1Pos[0] in curEntTypes[i].keys():
                    entType = curEntTypes[i][ent1Pos[0]]["type"]
                else: 
                    entType = "O"
                if entType in entTypeToInd.keys():
                    featureVector.append(entTypeToInd[entType])
                else: 
                    logging.error("Could not find entity type {} in entTypeToInd! This should not have happened!".format(entType))
                    exit(0)
                #RELATION-O
                if ent2Pos[0] in curEntTypes[i].keys():
                    entType = curEntTypes[i][ent2Pos[0]]["type"]
                else: 
                    entType = "O"
                if entType in entTypeToInd.keys():
                    featureVector.append(entTypeToInd[entType])
                else: 
                    logging.error("Could not find entity type {} in entTypeToInd! This should not have happened!".format(entType))
                    exit(0)
                #Capitalization
                #RELATION-S
                if len(curEntities[i][0]) and curEntities[i][0] == curEntities[i][0].upper():
                    featureVector.append(1)
                else: 
                    featureVector.append(0)
                #RELATION-O
                if len(curEntities[i][1]) and curEntities[i][1] == curEntities[i][1].upper():
                    featureVector.append(1)
                else: 
                    featureVector.append(0)
                #Contexts
                #POS tags
                #Lemmas
                leftPOS = []
                leftLemma = []
                rightPOS = []
                rightLemma = []
                for entPos in range(len(curEntityPos[i])):
                    for i in range(windowSize):
                        lPos = curEntityPos[i][entPos][0]-windowSize+i
                        post = "UNK"
                        lemma = "UNK"
                        if len(curPOS[i]) and lPos >= 0: 
                            post = curPOS[i][lPos]
                        if len(curLemmas[i]) and lPos >=0:
                            lemma = curLemmas[i][lPos]
                            if lemma not in lemmaToInd.keys():
                                lemma = "UNK"
                        if post not in posToInd.keys():
                            logging.error("Could not find POS tag {} in posToInd! This should not have happened!".format(post))
                            exit(0)
                        if lemma not in lemmaToInd.keys():
                            logging.error("Could not find lemma {} in lemmaToInd! This should not have happened!".format(lemma))
                            exit(0)
                        leftPOS.append(posToInd[post])
                        leftLemma.append(lemmaToInd[lemma])
                    for i in range(windowSize):
                        rPos = curEntityPos[i][entPos][1]+1+i
                        post = "UNK"
                        lemma = "UNK"
                        if len(curPOS[i]) and rPos < len(curPOS[i]): 
                            post = curPOS[i][rPos]
                        if len(curLemmas[i]) and rPos < len(curLemmas[i]):
                            lemma = curLemmas[i][rPos]
                            if lemma not in lemmaToInd.keys():
                                lemma = "UNK"
                        if post not in posToInd.keys():
                            logging.error("Could not find POS tag {} in posToInd! This should not have happened!".format(post))
                            exit(0)
                        if lemma not in lemmaToInd.keys():
                            logging.error("Could not find lemma {} in lemmaToInd! This should not have happened!".format(lemma))
                            exit(0)
                        rightPOS.append(posToInd[post])
                        rightLemma.append(lemmaToInd[lemma])
                featureVector.extend(leftPOS)
                featureVector.extend(rightPOS)
                featureVector.extend(leftLemma)
                featureVector.extend(rightLemma)

                rootNodes.append(newTree.getRootNode())
                newTree.to(device)
                trees.append(newTree)
                featureVectors.extend(featureVector)

            bTree = BatchedTree(trees)
                
            #Add code from testModel
            texts = [curSentences]
            bTrees = [bTree]
            rootNodes = [rootNodes]
            featureVectors = [featureVectors]
            rootNodes = torch.tensor(rootNodes).to(device)
            featureVectors = torch.tensor(featureVectors).to(device)

            outputs = model(
                bTrees=bTrees,
                rootNodes=rootNodes,
                featureVectors=featureVectors,
            )

            _, preds = torch.max(outputs, dim=1)

            logging.info(f"\tEntity Pair: ({Colours.OKBLUE}{pair[0]}{Colours.ENDC}, {Colours.HEADER}{pair[1]}{Colours.ENDC})")
            logging.info(f"\tPrediction: {Colours.BOLD}{model.labelToText(preds.cpu().numpy())[0]}{Colours.ENDC}")
            logging.info(f"\tTexts:")
            for i in range(len(curSentences)):
                if len(curSentences[i]) == 0:
                    continue
                ent1Pos = (curSentences[i].index(curEntities[i][0]), curSentences[i].index(curEntities[i][0])+len(curEntities[i][0])-1)
                ent2Pos = (curSentences[i].index(curEntities[i][1]), curSentences[i].index(curEntities[i][1])+len(curEntities[i][1])-1)
                if ent1Pos[0] < ent2Pos[0]:
                    curSentenceHighlighted = curSentences[i][:ent1Pos[0]] + Colours.OKBLUE + curSentences[i][ent1Pos[0]:ent1Pos[1]+1] + Colours.ENDC 
                    if ent2Pos[0] > ent1Pos[1]: #No overlap between entity mentions
                        curSentenceHighlighted += curSentences[i][ent1Pos[1]+1:ent2Pos[0]] + Colours.HEADER + curSentences[i][ent2Pos[0]:ent2Pos[1]+1] + Colours.ENDC + curSentences[i][ent2Pos[1]+1:]
                    elif ent2Pos[1] > ent1Pos[1]: #Some overlap but difference is non-NULL
                        curSentenceHighlighted += Colours.HEADER + curSentences[i][ent1Pos[1]+1:ent2Pos[1]+1] + Colours.ENDC + curSentences[i][ent2Pos[1]+1:]
                    else:   #One entity mention entirely inside the other entity mention
                        curSentenceHighlighted += curSentences[i][ent1Pos[1]+1:] 
                else: 
                    curSentenceHighlighted = curSentences[i][:ent2Pos[0]] + Colours.HEADER + curSentences[i][ent2Pos[0]:ent2Pos[1]+1] + Colours.ENDC 
                    if ent1Pos[0] > ent2Pos[1]: #No overlap between entity mentions
                        curSentenceHighlighted += curSentences[i][ent2Pos[1]+1:ent1Pos[0]] + Colours.OKBLUE + curSentences[i][ent1Pos[0]:ent1Pos[1]+1] + Colours.ENDC + curSentences[i][ent1Pos[1]+1:]
                    elif ent1Pos[1] > ent2Pos[1]: #Some overlap but difference is non-NULL
                        curSentenceHighlighted += Colours.OKBLUE + curSentences[i][ent2Pos[1]+1:ent1Pos[1]+1] + Colours.ENDC + curSentences[i][ent1Pos[1]+1:]
                    else:   #One entity mention entirely inside the other entity mention
                        curSentenceHighlighted += curSentences[i][ent2Pos[1]+1:] 
                logging.info(f"\t\tSentence: {curSentenceHighlighted}")
                logging.info("----------")
            logging.info(f"\t"+"*"*80)
    else:
        texts, predictions, predictionProbs, trueRelations = testModel(
            model,
            testDataLoader,
            device, 
            debug
        )

        if device != "cpu":
            if torch.is_tensor(predictions[0]):
                newPredictions = []
                for pred in predictions:
                    newPredictions.append(pred.cpu().tolist())
                predictions = newPredictions
            if torch.is_tensor(predictionProbs[0]):
                newPredProbs = []
                for predProb in predictionProbs:
                    newPredProbs.append(predProb.cpu().tolist())
                predictionProbs = newPredProbs
            if torch.is_tensor(trueRelations[0]):
                newTrueRelations = []
                for trueRel in trueRelations:
                    newTrueRelations.append(trueRel.cpu().tolist())
                trueRelations = newTrueRelations

        if savePredictions:
            with open("predictions_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(predictions, f)
            with open("predictionProbs_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(predictionProbs, f)
            with open("trueRelations_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(trueRelations, f)
            with open("getLabels_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(model.getLabels(), f)

        if printCorPredictions or printMisPredictions:
            labels = model.getLabels()
            corrInds = np.where(np.array(predictions)==np.array(trueRelations))
            for i in range(len(predictions)):
                if printCorPredictions:
                    if printMisPredictions:
                        condn = True
                    else:
                        condn = predictions[i] == trueRelations[i]
                else: 
                    condn = predictions[i] != trueRelations[i]
                if condn:
                    logging.info(f"\tTrue relation: {trueRelations[i]}")
                    logging.info(f"\tPrediction: {predictions[i]}")
                    logging.info(f"\tName: {labels[trueRelations[i]]}")
                    logging.info(f"\tTexts:")
                    for j in range(len(texts[i])):
                        logging.info(f"\t\tSentence: {texts[i][j]}")
                        logging.info("----------")
                    logging.info("*********************")

        if confusion:
            plotConfusion(predictions, trueRelations, model.getLabels(), debug)

        scores = computeF1score(predictions, trueRelations)
        logging.info("Results")
        logging.info("\tPrecision:")
        logging.info("\t\tMacro average: {}".format(round(scores["prec"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["prec"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["prec"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(model.labelToText([clas])[0], round(scores["prec"]["perClass"][clas],2)))
        logging.info("\tRecall:")
        logging.info("\t\tMacro average: {}".format(round(scores["rec"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["rec"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["rec"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(model.labelToText([clas])[0], round(scores["rec"]["perClass"][clas],2)))
        logging.info("\tF1 Score:")
        logging.info("\t\tMacro average: {}".format(round(scores["f1"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["f1"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["f1"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(model.labelToText([clas])[0], round(scores["f1"]["perClass"][clas],2)))

