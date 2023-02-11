from Modules.helper.imports.functionImports import checkFile, extractSentences, resolveCorefences, buildTerrierIndex
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
    "-wiki",
    help="Path to file containing wiki articles",
    required=True
)

parser.add_argument(
    "-invertedIndex",
    help="Path to store output Terrier index folder",
    required=True
)

parser.add_argument(
    "-docs",
    help="Path to store output documents file (extension='.pkl')",
    required=True
)



args = parser.parse_args()

debug = args.debug
logFile = args.log
wikiFile = args.wiki
terrierIndexPath = args.invertedIndex
docsFile = args.docs

checkFile(wikiFile, ".pkl")

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.DEBUG)


with open(wikiFile, 'rb') as f:
    wikiArticles = pickle.load(f)

allArticles = []
for k in wikiArticles.keys():
    allArticles.append(wikiArticles[k])

allArticles = list(set(allArticles))

allArticles = resolveCorefences(allArticles, debug)
allSentences = extractSentences(allArticles, debug)
docs = buildTerrierIndex(allSentences, terrierIndexPath, debug)

with open(docsFile.split(".")[0] + ".pkl", "wb") as f:
    pickle.dump(docs, f)