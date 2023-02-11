from Modules.helper.imports.functionImports import checkFile, createDataLoader, trainModel, evaluateModel
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
    "-train",
    help="Path to file containing training examples (extension=.pkl)",
    default="all_examples_2.pkl"
)

parser.add_argument(
    "-valid",
    help="Path to file containing validation examples (extension=.pkl)",
    default="all_examples_valid_2.pkl"
)

parser.add_argument(
    "-test",
    help="Path to file containing test  examples (extension=.pkl)",
    default="all_examples_test_2.pkl"
)

parser.add_argument(
    "-trainValTest",
    action="store_true",
    help="Boolean flag to split train set into train, validation and test set",
    default=False
)

parser.add_argument(
    "-histogram",
    action="store_true",
    help="Boolean flag to show histogram of examples"
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
    required=True
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for training",
    required=True
)

parser.add_argument(
    "-pretrainedModel",
    choices=["bert-base-uncased", "bert-base-cased"],
    help="Pretrained BERT model to use",
    default="bert-base-cased"
)

parser.add_argument(
    "-epochs",
    type=int,
    help="No. of epochs to train for",
    default=50
)

parser.add_argument(
    "-load",
    help="Path to file containing model to load",
    default=None
)

parser.add_argument(
    "-balance",
    action="store_true",
    help="Boolean flag to balance train dataset if not already balanced"
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
trainFile = args.train
validFile = args.valid
testFile = args.test
trainValTest = args.trainValTest
histogram = args.histogram
maxLen = args.maxLen
batchSize = args.batchSize
pretrainedModel = args.pretrainedModel
epochs = args.epochs
loadModel = args.load
learningRate = args.learningRate
balance = args.balance

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

checkFile(trainFile, ".pkl")
if not trainValTest:
    checkFile(validFile, ".pkl")
    checkFile(testFile, ".pkl")

with open(trainFile, "rb") as f:
    train = pickle.load(f)

if not trainValTest:
    with open(validFile, "rb") as f:
        valid = pickle.load(f)
    with open(testFile, "rb") as f:
        test = pickle.load(f)

df = pd.DataFrame(train)

if not trainValTest:
    df_train = df
    df_valid = pd.DataFrame(valid)
    df_test = pd.DataFrame(test)

if maxLen == None:
    maxLen = min(256, max([len(s) for s in df.text.to_numpy()]))

classes = np.unique(df.relation.to_numpy())

if histogram:
    if not trainValTest:
        sns.histplot([reln.split("/")[-1]for reln in df_train.relation.to_numpy()], stat="percent")
        plt.title("Train Data")
        plt.show()
        sns.histplot([reln.split("/")[-1]for reln in df_valid.relation.to_numpy()], stat="percent")
        plt.title("Validation Data")
        plt.show()
        sns.histplot([reln.split("/")[-1]for reln in df_test.relation.to_numpy()], stat="percent")
        plt.title("Test Data")
        plt.show()
    else:
        sns.histplot([reln.split("/")[-1]for reln in df.relation.to_numpy()], stat="percent")
        plt.title("Data")
        plt.show()

#Bert model expects integer targets, not strings
if not loadModel:
    le = preprocessing.LabelEncoder()
    df['relation'] = le.fit_transform(df['relation'])
    with open("labelEncoder.pkl","wb") as f:
        pickle.dump(le, f)

#Balance dataset 
if -balance:
    vals, counts = np.unique(df.relation.to_numpy(), return_counts=True)
    if not np.allclose(counts, [(sum(counts)/len(counts))]*len(vals)):
        logging.info("Dataframe not balanced!")
        majorVal = np.argmax(counts)
        dfMajor = df[df.relation == vals[majorVal]]
        numSamples = dfMajor.shape[0]
        for val in vals: 
            if val == vals[majorVal]:
                continue
            dfTemp = df[df.relation == val]
            dfTemp = resample(
                dfTemp, 
                replace=True, 
                n_samples=numSamples,
                random_state=26
            )
            dfMajor = pd.concat([dfMajor, dfTemp])
        df = dfMajor.copy()
        logging.info(f"Resampled dataframe to balance it")
        vals, counts = np.unique(df.relation.to_numpy(), return_counts=True)
        logging.info(f"vals: {vals}")
        logging.info(f"new counts: {counts}")
#Balance dataset

if not trainValTest:
    df_train = df
    df_valid = pd.DataFrame(valid)
    df_test = pd.DataFrame(test)
    if not loadModel:
        df_valid['relation'] = le.transform(df_valid['relation'])
        df_test['relation'] = le.transform(df_test['relation'])

tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)

if trainValTest:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
    df_valid, df_test = train_test_split(df_test, test_size=0.6, random_state=13)

if loadModel:
    model = torch.load(loadModel)
    if not trainValTest:
        df_train["relation"] = model.textToLabel(df_train["relation"])
        df_valid["relation"] = model.textToLabel(df_valid["relation"])
        df_test["relation"] = model.textToLabel(df_test["relation"])
    else: 
        df["relation"] = model.textToLabel(df["relation"])
else:
    model = RelationClassifier(len(classes), pretrainedModel, le)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
model = model.to(device)

trainDataLoader = createDataLoader(df_train, tokenizer, maxLen, batchSize, debug)
validDataLoader = createDataLoader(df_valid, tokenizer, maxLen, batchSize, debug)
testDataLoader = createDataLoader(df_test, tokenizer, maxLen, batchSize, debug)

optimizer = transformers.AdamW(
    model.parameters(), 
    lr=learningRate, 
    correct_bias=False, 
    weight_decay=0.01
)
totalSteps = len(trainDataLoader)*epochs
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=totalSteps
)
lossFunction = torch.nn.CrossEntropyLoss().to(device)

history = {
    "trainAcc": [],
    "trainLoss": [],
    "validAcc": [],
    "validLoss": [],
    "testAcc": [],
    "testLoss": []
}
bestAcc = 0 

for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs}")

    trainAcc, trainLoss = trainModel(
        model,
        trainDataLoader,
        lossFunction,
        optimizer, 
        device, 
        scheduler,
        len(df_train),
        debug
    )

    logging.info(f"\tTrain Accuracy: {trainAcc}, Train loss: {trainLoss}")

    validAcc, validLoss = evaluateModel(
        model,
        validDataLoader,
        lossFunction,
        device, 
        len(df_valid),
        debug
    )

    logging.info(f"\Validation Accuracy: {validAcc}, Validation loss: {validLoss}")

    testAcc, testLoss = evaluateModel(
        model,
        testDataLoader,
        lossFunction,
        device, 
        len(df_test),
        debug
    )

    logging.info(f"\tTest Accuracy: {testAcc}, Tess loss: {testLoss}")

    history["trainAcc"].append(trainAcc)
    history["trainLoss"].append(trainLoss)

    history["validAcc"].append(validAcc)
    history["validLoss"].append(validLoss)

    history["testAcc"].append(validAcc)
    history["testLoss"].append(validLoss)

    if validAcc > bestAcc: 
        torch.save(model.state_dict(), "model.pt")
        bestAcc = validAcc

with open("history.pkl","wb") as f:
    pickle.dump(history, f)