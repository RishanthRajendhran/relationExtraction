from Modules.helper.imports.functionImports import checkFile, createDataLoader, testModel, computeF1score, plotConfusion
from Modules.helper.imports.packageImports import argparse, pickle, logging, np, tf, hub, text, plt, train_test_split, transformers, sns, pd, torch, preprocessing
from Modules.helper.imports.classImports import RelExtDataset, RelationClassifier

tf.get_logger().setLevel("ERROR")

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
    "-numClasses",
    type=int,
    help="No. of classes the input model is built for",
    default=3
)

parser.add_argument(
    "-live",
    action="store_true",
    help="Boolean flag to enable live demo mode"
)

parser.add_argument(
    "-le",
    "--labelEncoder",
    help="Path to file containing label encoder object",
    default=None
    # default="labelEncoder.pkl"
)

parser.add_argument(
    "-example",
    help="Example sentence to test the model on in live demo mode",
    default=None
)

parser.add_argument(
    "-baseline",
    choices=["majority", "random"],
    help="Test Baseline models",
    default=None
)

parser.add_argument(
    "-plotConfusion",
    action="store_true",
    help="Boolean flag to plot confusion matrix after evaluation"
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
modelFile = args.model
testFile = args.test
maxLen = args.maxLen
batchSize = args.batchSize
pretrainedModel = args.pretrainedModel
numClasses = args.numClasses
live = args.live
labelEncoderFile = args.labelEncoder
example = args.example
baseline = args.baseline
plotConfusion = args.plotConfusion

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if live: 
    logging.basicConfig(level=logging.ERROR)
else:
    if labelEncoderFile:
        checkFile(labelEncoderFile, ".pkl")
        with open(labelEncoderFile, "rb") as f:
            le = pickle.load(f)

    checkFile(testFile, ".pkl")
    with open(testFile, "rb") as f:
        test = pickle.load(f)
    df = pd.DataFrame(test)
    if labelEncoderFile:
        #Bert model expects integer targets, not strings
        df['relation'] = le.fit_transform(df['relation'])

if baseline:
    if live:
        logging.error("Cannot test baseline models in live demo mode!")
        exit(0)
    if labelEncoderFile == None:
        logging.error("Label encoder needs to be provided for baseline evaluation!")
        exit(0)
    if baseline == "majority":
        labels, counts = np.unique(df.relation.to_numpy(),return_counts=True)
        logging.info("Baseline: Majority")
        majorityInd = np.argmax(counts)
        logging.info("\tMajority prediction: {}".format(le.inverse_transform([labels[majorityInd]])[0]))
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
            logging.info("\t\t\t{}: {}".format(le.inverse_transform([clas])[0], round(scores["prec"]["perClass"][clas],2)))
        logging.info("\tRecall:")
        logging.info("\t\tMacro average: {}".format(round(scores["rec"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["rec"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["rec"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(le.inverse_transform([clas])[0], round(scores["rec"]["perClass"][clas],2)))
        logging.info("\tF1 Score:")
        logging.info("\t\tMacro average: {}".format(round(scores["f1"]["macro"],2)))
        logging.info("\t\tMicro average: {}".format(round(scores["f1"]["micro"],2)))
        logging.info("\t\tPer Class:")
        for clas in scores["f1"]["perClass"].keys():
            logging.info("\t\t\t{}: {}".format(le.inverse_transform([clas])[0], round(scores["f1"]["perClass"][clas],2)))
else:
    if maxLen == None:
        if live: 
            maxLen = 256
        else:
            maxLen = min(256, max([len(s) for s in df.text.to_numpy()]))


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available():
        model = torch.load(modelFile)
    else:
        model = torch.load(modelFile, map_location={'cuda:0': 'cpu'})
    model = model.to(device)
    if not live: 
        df['relation'] = model.textToLabel(df['relation'])
    
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)
    if not live: 
        testDataLoader = createDataLoader(df, tokenizer, maxLen, batchSize, debug)

    if live: 
        if example == None:
            logging.error("Need to provide an example sentence using the -example flag when in live mode!")
            exit(0)
        encoding = tokenizer.encode_plus(
            example,
            max_length=maxLen,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True
        )
        modelInput = {
            "text": [example],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        outputs = model(
            input_ids=modelInput["input_ids"].to(device),
            attention_mask=modelInput["attention_mask"].to(device)
        )
        _, preds = torch.max(outputs, dim=1)
        logging.info("Input: {}".format(modelInput["text"][0]))
        logging.info(f"Prediction: {model.labelToText(preds.numpy())[0]}")
    else:
        texts, predictions, predictionProbs, trueRelations = testModel(
            model,
            testDataLoader,
            device, 
            len(df),
            debug
        )

        if plotConfusion:
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
