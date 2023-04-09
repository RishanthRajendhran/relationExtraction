import torch
import logging
from Modules.helper.classes.tree import Tree
from Modules.helper.classes.batchedTree import BatchedTree

class RelExtDataset:
    def __init__(self, texts, relation, entities, entityPos, bottomUpOrders, hierarchies, cleanedTexts, allWords, sentenceVectors, allPOS, allEntTypes, allLemmas, allShapes, embeddingSize, entTypeToInd, posToInd, lemmaToInd, maxSents, hiddenSize, windowSize, device):
        self.texts = texts 
        self.relation = relation 
        self.embeddingSize = embeddingSize
        self.entTypeToInd = entTypeToInd
        self.lemmaToInd = lemmaToInd
        self.posToInd = posToInd
        self.maxSents = maxSents
        self.hiddenSize = hiddenSize
        self.windowSize = windowSize
        self.entities = entities
        self.entityPos = entityPos
        self.bottomUpOrders = bottomUpOrders
        self.hierarchies = hierarchies
        self.cleanedTexts = cleanedTexts
        self.allWords = allWords
        self.sentenceVectors = sentenceVectors
        self.allPOS = allPOS
        self.allEntTypes = allEntTypes
        self.allLemmas = allLemmas
        self.allShapes = allShapes
        self.device = device

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        curSentences = self.texts[item]
        curEntities = self.entities[item]
        curEntTypes = self.allEntTypes[item]
        curPOS = self.allPOS[item]
        curLemmas = self.allLemmas[item]
        curEntityPos = self.entityPos[item]
        rootNodes = []
        trees = []
        featureVectors = []
        
        #Pad missing sentences upto self.maxSents
        for i in range(len(curSentences), self.maxSents):
            curSentences.append("")
            curEntities.append(("",""))
            curEntTypes.append({})
            curPOS.append([])
            curLemmas.append([])
            curEntityPos.append(((-1,-1), (-1,-1)))

        #Trim sentences to self.maxSents
        curSentences = curSentences[:self.maxSents]
        #Encode every sentence
        for i in range(len(curSentences)):
            #Build Tree
            newTree = Tree(self.hiddenSize, self.embeddingSize)
            if len(curSentences[i])==0:
                newTree.addNode()
            else:
                bottomUpOrder = self.bottomUpOrders[item][i]
                hierarchy = self.hierarchies[item][i]
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
                            curNodeID = newTree.addNode(parentID, torch.tensor(self.sentenceVectors[item][i][curNodePos], dtype=torch.float32))
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
            if entType in self.entTypeToInd.keys():
                featureVector.append(self.entTypeToInd[entType])
            else: 
                logging.error("Could not find entity type {} in entTypeToInd! This should not have happened!".format(entType))
                exit(0)
            #RELATION-O
            if ent2Pos[0] in curEntTypes[i].keys():
                entType = curEntTypes[i][ent2Pos[0]]["type"]
            else: 
                entType = "O"
            if entType in self.entTypeToInd.keys():
                featureVector.append(self.entTypeToInd[entType])
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
                for i in range(self.windowSize):
                    lPos = curEntityPos[i][entPos][0]-self.windowSize+i
                    post = "UNK"
                    lemma = "UNK"
                    if len(curPOS[i]) and lPos >= 0: 
                        post = curPOS[i][lPos]
                    if len(curLemmas[i]) and lPos >=0:
                        lemma = curLemmas[i][lPos]
                        if lemma not in self.lemmaToInd.keys():
                            lemma = "UNK"
                    if post not in self.posToInd.keys():
                        logging.error("Could not find POS tag {} in posToInd! This should not have happened!".format(post))
                        exit(0)
                    if lemma not in self.lemmaToInd.keys():
                        logging.error("Could not find lemma {} in lemmaToInd! This should not have happened!".format(lemma))
                        exit(0)
                    leftPOS.append(self.posToInd[post])
                    leftLemma.append(self.lemmaToInd[lemma])
                for i in range(self.windowSize):
                    rPos = curEntityPos[i][entPos][1]+1+i
                    post = "UNK"
                    lemma = "UNK"
                    if len(curPOS[i]) and rPos < len(curPOS[i]): 
                        post = curPOS[i][rPos]
                    if len(curLemmas[i]) and rPos < len(curLemmas[i]):
                        lemma = curLemmas[i][rPos]
                        if lemma not in self.lemmaToInd.keys():
                            lemma = "UNK"
                    if post not in self.posToInd.keys():
                        logging.error("Could not find POS tag {} in posToInd! This should not have happened!".format(post))
                        exit(0)
                    if lemma not in self.lemmaToInd.keys():
                        logging.error("Could not find lemma {} in lemmaToInd! This should not have happened!".format(lemma))
                        exit(0)
                    rightPOS.append(self.posToInd[post])
                    rightLemma.append(self.lemmaToInd[lemma])
            featureVector.extend(leftPOS)
            featureVector.extend(rightPOS)
            featureVector.extend(leftLemma)
            featureVector.extend(rightLemma)

            rootNodes.append(newTree.getRootNode())
            newTree.to(self.device)
            trees.append(newTree)
            featureVectors.extend(featureVector)

        bTree = BatchedTree(trees)
            
        return {
            "texts": curSentences,
            "bTree": bTree,
            "rootNodes": rootNodes,
            "targets": torch.tensor(self.relation[item], dtype=torch.long),
            "featureVectors": featureVectors,
        }