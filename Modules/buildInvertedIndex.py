from Modules.helper.imports.functionImports import checkFile, extractSentences, extractWords, resolveCorefences
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
    help="Path to output file (extension: .pkl) where inverted index is to be stored",
    required=True
)



args = parser.parse_args()

debug = args.debug
logFile = args.log
wikiFile = args.wiki
invertedIndexFile = args.invertedIndex

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

allArticles = resolveCorefences(allArticles, debug)
allSentences = extractSentences(allArticles, debug)
allWords = extractWords(allSentences, debug)

invertedIndex = {
    "sentences": allSentences,
    "index": {}
}

for word in allWords:
    if word not in invertedIndex["index"].keys():
        invertedIndex["index"][word] = []

for s in range(len(allSentences)):
    if debug:
        logging.info(f"Processing sentence {s}/{len(allSentences)}")
    wordsInSentence = extractWords([allSentences[s]])
    for word in wordsInSentence:
        if word in allWords:
            invertedIndex["index"][word].append(s)

with open(invertedIndexFile.split(".")[0] + ".pkl", "wb") as f: 
    pickle.dump(invertedIndex, f)