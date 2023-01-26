from Modules.helper.imports.functionImports import extractMappings, extractRelationInstances, checkFile, checkPath, getWikiArticles
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
    help="Path to txt validation file containing relation instances",
    default=dataConfig.dataSets["fb15k"]["validPath"]
)

parser.add_argument(
    "-test",
    help="Path to txt test file containing relation instances",
    default=dataConfig.dataSets["fb15k"]["testPath"]
)

parser.add_argument(
    "-load",
    action="store_true",
    help="Boolean flag to indicate that mappings and relation/entities can be loaded"
)

parser.add_argument(
    "-pickRelations",
    action="store_true",
    help="Boolean flag to enable pickRelations mode"
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


args = parser.parse_args()

debug = args.debug
logFile = args.log
mapFile = args.map 
load = args.load
pickRelations = args.pickRelations
trainFile = args.train 
validFile = args.valid 
testFile = args.test 
wiki = args.wiki
numSamples = args.numSamples
mid2nameFile = args.mid2name
entitiesFile = args.entities
relationsFile = args.relations
wikiArticlesFile = args.wikiArticles

if pickRelations and numSamples==None:
    logging.critical("Need to specify numSamples to indicate no. of relations to choose in pickRelations mode")

checkFile(mapFile, ".tsv")
checkFile(trainFile, ".txt")
checkFile(validFile, ".txt")
checkFile(testFile, ".txt")
if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.DEBUG)

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
        wikiArticles = getWikiArticles(numSamples)
        with open(f'{wikiArticlesFile.split(".pkl")[0]}_{numSamples}.pkl', 'wb') as f:
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
    elif pickRelations:
        checkFile(entitiesFile, ".pkl")
        checkFile(relationsFile, ".pkl")

        with open(entitiesFile, 'rb') as f:
            entities = pickle.load(f)

        with open(relationsFile, 'rb') as f:
            relations = pickle.load(f)

        if numSamples > len(relations):
            logging.critical("Cannot sample more relations than available!")

        chosenRelations = np.random.choice(len(relations), numSamples, replace=False)

        newRelations = {}
        newEntities = {}
        relationsKeys = list(relations.keys())
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

        with open(f'{entitiesFile.split(".pkl")[0]}_{numSamples}.pkl', 'wb') as f:
            pickle.dump(newEntities, f)

        with open(f'{relationsFile.split(".pkl")[0]}_{numSamples}.pkl', 'wb') as f:
            pickle.dump(newRelations, f)
    else:
        mid2name = extractMappings(mapFile, debug)
        entities, relations = extractRelationInstances(trainFile, debug)

        with open(mid2nameFile, 'wb') as f:
            pickle.dump(mid2name, f)

        with open(entitiesFile, 'wb') as f:
            pickle.dump(entities, f)

        with open(relationsFile, 'wb') as f:
            pickle.dump(relations, f)