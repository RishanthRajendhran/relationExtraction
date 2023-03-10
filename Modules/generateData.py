from Modules.helper.imports.functionImports import extractMappings, extractRelationInstances, checkFile, checkPath, getWikiArticles, getWikiSummaries
from Modules.helper.imports.packageImports import argparse, sys, pickle, logging, np
from Modules.helper.imports.configImports import dataConfig

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
    "-map",
    help="Path to TSV file containing mappings between MIDs and wikipedia titles",
    default=dataConfig.dataSets["mid2name"]["mapPath"]
)

parser.add_argument(
    "-train",
    help="Path to txt train file containing relation instances",
    default=dataConfig.dataSets["fb15k"]["trainPath"]
)

parser.add_argument(
    "-valid",
    help="Path to txt train file containing relation instances",
    default=dataConfig.dataSets["fb15k"]["validPath"]
)

parser.add_argument(
    "-test",
    help="Path to txt train file containing relation instances",
    default=dataConfig.dataSets["fb15k"]["testPath"]
)

parser.add_argument(
    "-mode",
    choices=["train", "test", "valid"],
    help="Used to indicate the type of file being worked on (mappings would be extracted from mid2name only in train mode)",
    required=True
)

parser.add_argument(
    "-load",
    action="store_true",
    help="Boolean flag to indicate that mappings and relation/entities can be loaded"
)

parser.add_argument(
    "-pickRelations",
    type=str,
    help="Flag to enable pickRelations mode and specify no. of relations to pick/text file containing relations to pick one per line",
    default=None,
)

parser.add_argument(
    "-topK",
    action="store_true",
    help="Boolean flag to enable picking the top K relations based on counts in pickRelations mode",
)

parser.add_argument(
    "-wiki",
    action="store_true",
    help="Boolean flag to enable wiki mode"
)

parser.add_argument(
    "-numSamples",
    type=int,
    default=None,
    help="No. of relations sampled (Used for file naming purposes)"
)

parser.add_argument(
    "-mid2name",
    help="Path to file containing mid2name dictionary",
    default="mid2name.pkl"
)

parser.add_argument(
    "-entities",
    help="Path to file containing entities dictionary",
    default="entities.pkl"
)

parser.add_argument(
    "-relations",
    help="Path to file containing relations dictionary",
    default="relations.pkl"
)

parser.add_argument(
    "-wikiArticles",
    help="Path to file containing wiki articles list",
    default="wikiArticles.pkl"
)

parser.add_argument(
    "-article",
    action="store_true",
    help="Boolean flag to be used in wiki mode to generate articles instead of summaries"
)

parser.add_argument(
    "-maxInstsPerRel",
    type=int,
    help="Max. no. of instances per relation in pickRelation mode",
    default=np.inf
)

parser.add_argument(
    "-random",
    action="store_true",
    help="Boolean flag to be used in wiki mode to generate random articles",
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
mapFile = args.map 
load = args.load
pickRelations = args.pickRelations
trainFile = args.train 
validFile = args.valid 
testFile = args.test
mode = args.mode
wiki = args.wiki
numSamples = args.numSamples
mid2nameFile = args.mid2name
entitiesFile = args.entities
relationsFile = args.relations
wikiArticlesFile = args.wikiArticles
article = args.article
maxInstsPerRel = args.maxInstsPerRel
random = args.random
topK = args.topK

checkFile(mapFile, ".tsv")
if mode == "train":
    checkFile(trainFile, ".txt")
elif mode == "valid":
    checkFile(validFile, ".txt")
elif mode == "test":
    checkFile(testFile, ".txt")

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if wiki:
    if load:
        checkFile(wikiArticlesFile, ".pkl")
        with open(f'{wikiArticlesFile.split(".pkl")[0]}_{numSamples}.pkl', 'rb') as f:
            wikiArticles = pickle.load(f)
        print(f"No. of wikiArticles: {len(wikiArticles)}")
        print(f"Sample article titles:")
        wikiTitles = list(wikiArticles.keys())
        for i in range(10):
            print(f"\t{wikiTitles[i]}")
    else:
        if article and random: 
            wikiArticles = getWikiArticles(numSamples, debug)
        else: 
            checkFile(mid2nameFile, ".pkl")
            checkFile(entitiesFile, ".pkl")

            with open(mid2nameFile, 'rb') as f:
                mid2name = pickle.load(f)

            with open(entitiesFile, 'rb') as f:
                entities = pickle.load(f)

            entsList = []
            for e in entities.keys():
                if e not in entsList:
                    entsList.append(e)
                for r in entities[e].keys():
                    for f in entities[e][r]:
                        if f not in entsList:
                            entsList.append(f)

            wikiArticles = getWikiSummaries(entsList, mid2name, article, debug)
            
        fileName = wikiArticlesFile.split(".pkl")[0]
        if random and numSamples:
            fileName += f'_{numSamples}.pkl'
        else: 
            fileName += ".pkl"
        with open(fileName, 'wb') as f:
            pickle.dump(wikiArticles, f)
else:
    if load:
        checkFile(mid2nameFile, ".pkl")
        checkFile(entitiesFile, ".pkl")
        checkFile(relationsFile, ".pkl")

        with open(mid2nameFile, 'rb') as f:
            mid2name = pickle.load(f)

        with open(entitiesFile, 'rb') as f:
            entities = pickle.load(f)

        with open(relationsFile, 'rb') as f:
            relations = pickle.load(f)
        
        if debug:
            for e in entities.keys():
                if e not in mid2name.keys():
                    logging.warning(f"Mapping for {e} missing!")
                    continue
                for r in entities[e].keys():
                    for f in entities[e][r]:
                        if f not in mid2name.keys():
                            logging.warning(f"Mapping for {f} missing!")
                            continue
                        e1 = mid2name[e][0]
                        e2 = mid2name[f][0]
                        logging.info("{:<25} {:<100} {:<25}".format(e1, r, e2))
        relnCounts = []
        for reln in relations.keys():
            relnCounts.append((len(relations[reln]), reln))
        relnCounts.sort()
        topK = 20
        logging.info(f"Top {topK} relations by counts:")
        for rel in relnCounts[-1:-(topK+1):-1]:
            logging.info(f"\t{rel[1]} ({rel[0]})")
    elif pickRelations:
        checkFile(entitiesFile, ".pkl")
        checkFile(relationsFile, ".pkl")

        with open(entitiesFile, 'rb') as f:
            entities = pickle.load(f)

        with open(relationsFile, 'rb') as f:
            relations = pickle.load(f)

        if pickRelations.isdigit():
            numSamples = int(pickRelations)
        else:
            checkFile(pickRelations, ".txt")
            with open(pickRelations, 'r') as f:
                relns = list(f.readlines())
                relns = [reln.replace("\n","") for reln in relns]
            
            curRels = {}
            for reln in relns:
                if reln not in relations.keys():
                    if debug:
                        logging.warning(f"{reln} not a recognized relation.")
                    continue
                chosenInstances = np.random.choice(len(relations[reln]), min(maxInstsPerRel, len(relations[reln])), replace=False)
                curRels[reln] = (np.array(relations[reln])[chosenInstances]).tolist()
            relations = curRels
            numSamples = len(relations)

        if numSamples > len(relations):
            logging.critical("Cannot sample more relations than available!")

        relationsKeys = list(relations.keys())
        if topK:
            relLens = []
            for rel in relationsKeys:
                relLens.append((len(relations[rel]), relationsKeys.index(rel)))
            relLens.sort(reverse=True)
            chosenRelations = [rl[1] for rl in relLens[:numSamples]]
        else:
            chosenRelations = np.random.choice(len(relations), numSamples, replace=False)

        newRelations = {}
        newEntities = {}
        for i in chosenRelations:
            if relationsKeys[i] not in newRelations.keys():
                newRelations[relationsKeys[i]] = []
            newRelations[relationsKeys[i]] = relations[relationsKeys[i]]
            for (e1, e2) in relations[relationsKeys[i]]:
                if e1 not in newEntities.keys():
                    newEntities[e1] = {}
                if relationsKeys[i] not in newEntities[e1].keys():
                    newEntities[e1][relationsKeys[i]] = []
                newEntities[e1][relationsKeys[i]].append(e2)
        if debug:
            logging.info(f"Extracted {len(newEntities)} entities for {len(newRelations)} relations.")

        with open(f'{entitiesFile.split(".pkl")[0]}_{numSamples}.pkl', 'wb') as f:
            pickle.dump(newEntities, f)

        with open(f'{relationsFile.split(".pkl")[0]}_{numSamples}.pkl', 'wb') as f:
            pickle.dump(newRelations, f)
    else:
        if mode == "train":
            mid2name = extractMappings(mapFile, debug)
            fileName = trainFile
        elif mode == "valid":
            fileName = validFile
        elif mode == "test":
            fileName = testFile
            
        entities, relations = extractRelationInstances(fileName, debug)

        if mode == "train":
            with open(mid2nameFile.split(".")[0]+".pkl", 'wb') as f:
                pickle.dump(mid2name, f)

        with open(entitiesFile.split(".")[0]+".pkl", 'wb') as f:
            pickle.dump(entities, f)

        with open(relationsFile.split(".")[0]+".pkl", 'wb') as f:
            pickle.dump(relations, f)