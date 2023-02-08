from Modules.helper.imports.functionImports import checkFile, extractPOStags, findWordInWords, extractWordsInBetween, extractWordsInWindow, extractNERtags, getShortestDependencyPath
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
    help="Path to file containing examples",
    default="examples_2.pkl"
)

parser.add_argument(
    "-out",
    help="Path to store feature vector of examples (extension=.pkl)",
    default="features_2.pkl"
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
mid2nameFile = args.mid2name
examplesFile = args.examples
outFile = args.out

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

windowSize = 3

checkFile(examplesFile, ".pkl")
checkFile(mid2nameFile, ".pkl")

with open(examplesFile, "rb") as f:
    examples = pickle.load(f)

with open(mid2nameFile, "rb") as f:
    mid2name = pickle.load(f)

np.random.shuffle(examples)

for ex in examples:
    relation = ex["relation"]
    e1 = ex["entities"][0]
    e2 = ex["entities"][1]
    sent = ex["text"]
    entity1 = ex["entPair"][0]
    entity2 = ex["entPair"][1]
    POStags, words = extractPOStags(sent, debug)
    ent1Start, ent1End =  findWordInWords(words, entity1, debug)
    ent2Start, ent2End =  findWordInWords(words, entity2, debug)
    if  ent1Start > ent2End:
        #Swap entities
        e1, e2 = e2, e1
        entity1, entity2 = entity2, entity1
        ent1Start, ent2Start = ent2Start, ent1Start
        ent1End, ent2End = ent2End, ent1End
    wordsInBetween = extractWordsInBetween(words, ent1End, ent2Start, debug)
    NERtags = extractNERtags(sent, debug)
    sdp, headDepPairs = getShortestDependencyPath(sent, words[ent1Start], words[ent2End],debug)
    #Feature engineering
    e1NER = "UNK"
    e2NER = "UNK" 
    for p in NERtags:
        if p[0] == entity1:
            e1NER = p[1]
        elif p[0] == entity2:
            e2NER = p[1]
    features = []
    lexicalMiddle = ""
    for w in wordsInBetween:
        lexicalMiddle += "{}/{} ".format(POStags[w][1], POStags[w][0])
    lexicalMiddle = lexicalMiddle.strip()

    syntacticMiddle = ""
    wordsLower = [w.lower() for w in words]
    #Ignore entity1 and entity2 in returned SDP
    startInd = ent1End - ent1Start + 1
    endInd = ent2End - ent2Start + 1
    for w in range(startInd, len(sdp)-endInd):
        curWord = sdp[w]
        prevWord = sdp[w-1]
        nextWord = sdp[w+1]
        wordLeft = ""
        if (prevWord, curWord) in headDepPairs.keys():
            wordLeft = "/{}".format(headDepPairs[(prevWord, curWord)])
        elif (curWord, prevWord) in headDepPairs.keys():
            wordLeft = "\{}".format(headDepPairs[(curWord, prevWord)])
        syntacticMiddle += wordLeft + " " + curWord + " "
        if w == (len(sdp)-endInd-1):
            wordRight = ""
            if (curWord, nextWord) in headDepPairs.keys():
                wordRight = "\{}".format(headDepPairs[(curWord, nextWord)])
            elif (nextWord, curWord) in headDepPairs.keys():
                wordRight = "/{}".format(headDepPairs[(nextWord, curWord)])
            syntacticMiddle += wordRight + " "
    syntacticMiddle = syntacticMiddle.strip()
    #Feature Type: Lexical 
    #Window size = 0 to windowSize
    #Feature type, left window, NE1, Middle, NE2, right window 
    for winSize in range(windowSize):
        wordsLeft1, _ = extractWordsInWindow(words, ent1Start, ent1End, winSize, debug)
        wordsLeft1 = [w if w == "#PAD#" else words[w] for w in wordsLeft1]
        _, wordsRight2 = extractWordsInWindow(words, ent2Start, ent2End, winSize, debug)
        wordsRight2 = [w if w == "#PAD#" else words[w] for w in wordsRight2]
        curFeature = "Lexical\t"
        curFeature += str(wordsLeft1) + "\t"
        curFeature += e1NER + "\t" 
        curFeature += lexicalMiddle + "\t"
        curFeature += e2NER + "\t"
        curFeature += str(wordsRight2) + "\t"
        features.append(curFeature)
    #Feature Type: Syntactic 
    #Window size = 0 to windowSize
    #Feature type, left window, NE1, Middle, NE2, right window 
    for winSize in range(windowSize):
        wordsLeft1, _ = extractWordsInWindow(words, ent1Start, ent1End, winSize, debug)
        for w in range(len(wordsLeft1)-1):
            if wordsLeft1[w] != "#PAD#":
                if (wordsLeft1[w], wordsLeft1[w+1]) in headDepPairs.keys():
                    wordsLeft1[w] = words[wordsLeft1[w]] + " /" + headDepPairs[(wordsLeft1[w], wordsLeft1[w+1])]
                else:
                    wordsLeft1[w] = words[wordsLeft1[w]]
        if len(wordsLeft1) and wordsLeft1[-1] != "#PAD#":
            if (wordsLeft1[-1], words[ent1Start]) in headDepPairs.keys():
                wordsLeft1[-1] = words[wordsLeft1[-1]] + " /" + headDepPairs[(wordsLeft1[-1], words[ent1Start])]
            else: 
                wordsLeft1[-1] = words[wordsLeft1[-1]]
        _, wordsRight2 = extractWordsInWindow(words, ent2Start, ent2End, winSize, debug)
        if len(wordsRight2) and wordsRight2[0] != "#PAD#":
            if (words[ent2End], wordsRight2[0]) in headDepPairs.keys():
                wordsRight2[0] =  "/" + headDepPairs[(words[ent2End], wordsRight2[0])] + " " + words[wordsRight2[0]]
            else: 
                wordsRight2[0] = words[wordsRight2[0]]
        for w in range(1,len(wordsRight2)):
            if wordsRight2[w] != "#PAD#":
                if (wordsRight2[w-1], wordsRight2[w]) in headDepPairs.keys():
                    wordsRight2[w] =  "/" + headDepPairs[(wordsRight2[w-1], wordsRight2[w])] + " " + words[wordsRight2[w]]
                else:
                    wordsRight2[w] = words[wordsRight2[w]]
        curFeature = "Syntactic\t"
        curFeature += str(wordsLeft1) + "\t"
        curFeature += e1NER + "\t" 
        curFeature += syntacticMiddle + "\t"
        curFeature += e2NER + "\t"
        curFeature += str(wordsRight2) + "\t"
        print(curFeature)
        features.append(curFeature)
