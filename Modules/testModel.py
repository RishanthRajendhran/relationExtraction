from Modules.helper.imports.functionImports import checkFile, createDataLoader, testModel, computeF1score, plotConfusion, extractNERtags, maskWordWithNER
from Modules.helper.imports.packageImports import argparse, pickle, logging, np, plt, train_test_split, transformers, sns, pd, torch, preprocessing, itertools, re
from Modules.helper.imports.classImports import RelExtDataset, RelationClassifier

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
    default="fullModel.pt"
)

parser.add_argument(
    "-test",
    help="Path to file containing test examples (extension=.pkl)",
    default="all_examples_test_2.pkl"
)

parser.add_argument(
    "-maxLen",
    type=int,
    help="Maximum length of input tokens (tokenizer)",
    default= None
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size for dataloader",
    default=8
)

parser.add_argument(
    "-pretrainedModel",
    choices=["bert-base-uncased", "bert-base-cased"],
    help="Pretrained BERT model to use",
    default="bert-base-cased"
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
    help="Example sentences to test the model on in live demo mode",
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
    "-printPredictions",
    action="store_true",
    help="Boolean flag to print predictions"
)

parser.add_argument(
    "-baseline",
    choices=["majority", "random"],
    help="Test Baseline models",
    default=None
)

args = parser.parse_args()

debug = args.debug
logFile = args.log
modelFile = args.model
testFile = args.test
maxLen = args.maxLen
batchSize = args.batchSize
pretrainedModel = args.pretrainedModel
live = args.live
examples = args.examples
confusion = args.confusion
maxSents = args.maxSents
histogram = args.histogram
entities = args.entities
savePredictions = args.savePredictions
printPredictions = args.printPredictions
baseline = args.baseline 

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if live: 
    logging.basicConfig(level=logging.ERROR)
else:
    checkFile(testFile, ".pkl")
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
        plt.show()
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
        plt.show()
        plt.clf()
    # #Bert model expects integer targets, not strings
    # df['relation'] = le.fit_transform(df['relation'])

if maxLen == None:
    if live: 
        maxLen = 256
    else:
        maxLen = min(256, max([len(s) for s in df.text.to_numpy()]))

tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)
# newNERtokens = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
newNERtokens = ["<RELATION-S>", "<RELATION-O>"]
newNERtokens = set(newNERtokens) - set(tokenizer.vocab.keys())
tokenizer.add_tokens(list(newNERtokens))

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

if not live: 
    df['relation'] = model.textToLabel(df['relation'])

if not live: 
    testDataLoader = createDataLoader(df, tokenizer, maxLen, maxSents, batchSize, debug)
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
            logging.info("\t{}".format(ex))
        for pair in itertools.permutations(entitiesUnderConsideration,2):
            curSentences = []
            sentInputIDs = []
            sentAttentionMasks = []
            entPairInds = []
            for ex in examples:
                if pair[0] in ex and pair[1] in ex:
                    modExample = maskWordWithNER(ex, pair[0], "RELATION", "S", debug)
                    modExample = maskWordWithNER(modExample, pair[1], "RELATION", "O", debug)
                    curSentences.append(modExample)
            if len(curSentences) == 0:
                continue
            #Pad missing sentences upto self.max_sents
            for i in range(len(curSentences), maxSents):
                curSentences.append("")
            #Trim sentences to self.max_sents
            curSentences = curSentences[:maxSents]
            #Encode every sentence
            for sentence in curSentences:
                encoding = tokenizer.encode_plus(
                    sentence,
                    max_length=maxLen,
                    add_special_tokens=True,
                    padding="max_length",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="pt",
                    truncation = True
                )
                sentInputIDs.append(encoding["input_ids"].reshape(-1,).tolist())
                sentAttentionMasks.append(encoding["attention_mask"].reshape(-1,).tolist())

                subjInd = 0
                objInd = 0
                if len(sentence):
                    subWords = tokenizer.convert_ids_to_tokens(encoding["input_ids"].reshape(-1,))
                    subj = re.compile("<[A-Za-z_]{1,}-S>")
                    subjMatches = list(filter(subj.match, subWords))
                    if len(subjMatches) == 0:
                        logging.debug(f"Could not find subject entity marker in {subWords}!")
                    elif len(subjMatches) > 1:
                        logging.debug(f"More than one subject entity marker found in  {subWords}!")
                    else:
                        subjInd =  subWords.index(subjMatches[0])

                    obj = re.compile("<[A-Za-z_]{1,}-O>")
                    objMatches = list(filter(obj.match, subWords))
                    if len(objMatches) == 0:
                        logging.debug(f"Could not find object entity marker in {subWords}!")
                    elif len(objMatches) > 1:
                        logging.debug(f"More than one object entity marker found in {subWords}!")
                    else:
                        objInd =  subWords.index(objMatches[0])
                
                entPairInd = (subjInd, objInd)
                entPairInds.append(entPairInd)

            sentInputIDs = torch.tensor([sentInputIDs])
            sentAttentionMasks = torch.tensor([sentAttentionMasks])
            entPairInds = torch.tensor([entPairInds])
            modelInput = {
                "texts": [curSentences],
                "input_ids": sentInputIDs,
                "attention_mask": sentAttentionMasks,
                "entity_pair_inds": entPairInds,
            }

            outputs, attnOuts = model(
                input_ids=modelInput["input_ids"].to(device),
                attention_mask=modelInput["attention_mask"].to(device),
                entity_pair_inds=modelInput["entity_pair_inds"].to(device),
                return_attn_out=True
            )
            _, preds = torch.max(outputs, dim=1)
            attnOuts = torch.sum(attnOuts, dim=2)
            attnOuts = attnOuts.cpu().detach().numpy()
            logging.info(f"\tEntity Pair: ({pair[0]}, {pair[1]}) as (RELATION-S, RELATION-O)")
            logging.info(f"\tPrediction: {model.labelToText(preds.cpu().numpy())[0]}")
            logging.info(f"\tTexts:")
            for i in range(len(curSentences)):
                if len(curSentences[i]) == 0:
                    continue
                logging.info(f"\t\tAttention weight: {(attnOuts[0][i]/sum(attnOuts[0])):0.2f}")
                logging.info(f"\t\tSentence: {curSentences[i]}")
                logging.info("----------")
            logging.info(f"\t*********")
    else:
        texts, predictions, predictionProbs, trueRelations, attnOuts = testModel(
            model,
            testDataLoader,
            device, 
            len(df),
            debug
        )

        attnOuts = torch.sum(attnOuts, dim=2)
        if savePredictions:
            with open("predictions_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(predictions, f)
            with open("attnOuts_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(attnOuts, f)
            with open("trueRelations_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(trueRelations, f)
            with open("getLabels_{}.pkl".format(testFile.split(".")[0]),"wb") as f: 
                pickle.dump(model.getLabels(), f)

        if printPredictions:
            labels = model.getLabels()
            corrInds = np.where(np.array(predictions)==np.array(trueRelations))
            for i in range(len(predictions)):
                if predictions[i] == trueRelations[i]:
                    logging.info(f"\tTrue relation: {trueRelations[i]}")
                    logging.info(f"\tPrediction: {predictions[i]}")
                    logging.info(f"\tName: {labels[trueRelations[i]]}")
                    logging.info(f"\tTexts:")
                    for j in range(len(texts[i])):
                        logging.info(f"\t\tAttention weight: {(attnOuts[i][j]/sum(attnOuts[i])):0.2f}")
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

