from Modules.helper.imports.functionImports import extractMappings, extractRelationInstances, checkFile, checkPath
from Modules.helper.imports.packageImports import argparse, sys, pickle, logging
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


args = parser.parse_args()

debug = args.debug
logFile = args.log
mapFile = args.map 
load = args.load
trainFile = args.train 
validFile = args.valid 
testFile = args.test 

checkFile(mapFile)
checkFile(trainFile)
checkFile(validFile)
checkFile(testFile)
if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.DEBUG)

if load:
    with open('mid2name.pkl', 'rb') as f:
        mid2name = pickle.load(f)

    with open('entities.pkl', 'rb') as f:
        entities = pickle.load(f)

    with open('relations.pkl', 'rb') as f:
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
else:
    mid2name = extractMappings(mapFile, debug)
    entities, relations = extractRelationInstances(trainFile, debug)

    with open('mid2name.pkl', 'wb') as f:
        pickle.dump(mid2name, f)

    with open('entities.pkl', 'wb') as f:
        pickle.dump(entities, f)

    with open('relations.pkl', 'wb') as f:
        pickle.dump(relations, f)